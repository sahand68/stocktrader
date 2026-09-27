import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trader.backtest import Costs
from trader.data import bar_minutes, load_bars, make_exchange, periods_per_year
from trader.jev import JevOracle
from trader.paper import KillSwitch, PaperTrader
from trader.strategies import BuyAndHold, JevDirection, SmaTrend
from trader.trials import TrialLedger
from trader.validate import Report, validate

STRATEGIES = ("jev", "sma-trend", "buy-and-hold")


def make_strategy(name: str, interval: str, model: str, allow_short: bool):
    if name == "jev":
        return JevDirection(JevOracle(model=model), interval, allow_short=allow_short)
    if name == "sma-trend":
        return SmaTrend(allow_short=allow_short)
    return BuyAndHold()


def placebo_bars(bars: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A random walk with the real data's length, volatility and volumes: nothing to find."""
    rng = np.random.default_rng(seed)
    vol = bars["close"].pct_change().std()
    close = bars["close"].iloc[0] * np.exp(np.cumsum(rng.normal(0, vol, len(bars))))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, vol / 2, len(bars))) * close
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
        "volume": bars["volume"].to_numpy(),
    }, index=bars.index)


def pct(x: float) -> str:
    return f"{x:+.1%}"


def print_report(report: Report) -> None:
    wf, m, bh, dsr = report.walk_forward, report.oos, report.buy_and_hold, report.dsr
    verdict = "APPROVED for paper trading" if report.approved else "REJECTED"
    print(f"\n{report.strategy}: {verdict}\n")
    for gate, ok in report.gates.items():
        print(f"  [{'x' if ok else ' '}] {gate}")

    print("\nOut of sample          strategy   buy-and-hold")
    print(f"  Sharpe               {m['sharpe']:8.2f}   {bh['sharpe']:8.2f}")
    print(f"  CAGR                 {pct(m['cagr']):>8}   {pct(bh['cagr']):>8}")
    print(f"  Max drawdown         {pct(m['max_drawdown']):>8}   {pct(bh['max_drawdown']):>8}")
    print(f"  Longest drawdown     {m['longest_drawdown_days']:6.0f} d   {bh['longest_drawdown_days']:6.0f} d")
    print(f"  Positive folds       {wf.positive_folds}/{len(wf.folds)}")

    noise = (f"the best of {dsr.n_trials} no-edge variations would show Sharpe {dsr.noise_benchmark:.2f}"
             if dsr.n_trials > 1 else "1 variation counted so far, so nothing to deflate yet")
    print(f"\nDeflated Sharpe: {dsr.probability:.3f} (needs > 0.95); {noise}.")

    print("\nCritic")
    for c in report.checks:
        print(f"  {c.status:4}  {c.name:18} {c.detail}")

    cols = [c for c in wf.folds.columns if c not in ("end", "calmar", "total_return")]
    print("\nWalk-forward folds (judge the worst one, not the average)")
    print(wf.folds[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nBy regime")
    print(report.regimes.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nParameters for paper trading: {wf.live_params}")


def cmd_fetch(args) -> None:
    bars = load_bars(args.exchange, args.symbol, args.interval, args.days, refresh=args.refresh)
    print(f"{len(bars)} {args.interval} bars of {args.symbol} from {bars.index[0]} to {bars.index[-1]}")


def cmd_backtest(args) -> None:
    bars = load_bars(args.exchange, args.symbol, args.interval, args.days)
    if args.placebo:
        bars = placebo_bars(bars)
    per_day = 1440 / bar_minutes(args.interval)
    config = {
        "strategy": args.strategy, "exchange": args.exchange, "symbol": args.symbol,
        "interval": args.interval, "days": args.days, "train_days": args.train_days,
        "test_days": args.test_days, "fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps,
        "allow_short": args.allow_short, "model": args.model, "placebo": args.placebo,
    }
    dataset = {k: config[k] for k in ("exchange", "symbol", "interval", "placebo")}
    n_trials = TrialLedger().record(dataset, config) + args.prior_trials

    strategy = make_strategy(args.strategy, args.interval, args.model, args.allow_short)
    report = validate(
        bars, strategy, Costs(args.fee_bps, args.slippage_bps), args.interval,
        periods_per_year(args.interval), int(args.train_days * per_day), int(args.test_days * per_day),
        n_trials, config,
    )
    print_report(report)
    if isinstance(strategy, JevDirection):
        o = strategy.oracle
        print(f"\nJev: {o.calls} new calls, {o.input_tokens:,} input tokens, ${o.cost_usd:.4f}; the rest came from cache")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"{args.strategy}_{args.symbol.replace('/', '-')}_{args.interval}_{stamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=float))
    print(f"Report: {path}")


def cmd_paper(args) -> None:
    report = json.loads(args.report.read_text())
    if not report["approved"] and not args.force:
        sys.exit("This strategy was rejected by its backtest. Paper trading it teaches nothing; "
                 "pass --force to run it anyway.")
    if report["config"]["placebo"]:
        sys.exit("That report is from placebo data.")
    c = report["config"]
    trader = PaperTrader(
        client=make_exchange(c["exchange"]),
        symbol=c["symbol"],
        interval=c["interval"],
        strategy=make_strategy(c["strategy"], c["interval"], c["model"], c["allow_short"]),
        params=report["live_params"],
        costs=Costs(c["fee_bps"], c["slippage_bps"]),
        kill=KillSwitch.from_report(report),
        periods_per_year=periods_per_year(c["interval"]),
        state_path=args.state_dir / "account.json",
        journal_path=args.state_dir / "journal.jsonl",
        capital=args.capital,
    )
    print(f"Paper trading {c['strategy']} on {c['symbol']} {c['interval']} with {report['live_params']}; "
          f"kill switch: {trader.kill}")
    trader.run(max_bars=args.bars, on_entry=lambda e: print(json.dumps(e, default=float)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trader", description="Crypto signals from TypeSafe Jev, validated before they trade.")
    sub = parser.add_subparsers(required=True)

    def market(p):
        p.add_argument("--exchange", default="binance", help="ccxt exchange id (binanceus / coinbaseexchange in the US)")
        p.add_argument("--symbol", default="BTC/USDT")
        p.add_argument("--interval", default="1h", choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        p.add_argument("--days", type=int, default=730)

    p = sub.add_parser("fetch", help="download and cache bars")
    market(p)
    p.add_argument("--refresh", action="store_true", help="ignore the cache")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("backtest", help="walk-forward test a strategy through the three gates")
    market(p)
    p.add_argument("--strategy", default="jev", choices=STRATEGIES)
    p.add_argument("--train-days", type=float, default=180)
    p.add_argument("--test-days", type=float, default=30)
    p.add_argument("--fee-bps", type=float, default=10.0, help="per side")
    p.add_argument("--slippage-bps", type=float, default=5.0, help="per side")
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--model", default="jev-latest", help="pin a version, e.g. jev-1.13.0, once thresholds matter")
    p.add_argument("--prior-trials", type=int, default=0,
                   help="variations tried on this data outside this tool; understating it only fools you")
    p.add_argument("--placebo", action="store_true", help="run on a random walk with the same volatility; should fail")
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("paper", help="paper trade an approved backtest report on live data")
    p.add_argument("report", type=Path)
    p.add_argument("--capital", type=float, default=10_000)
    p.add_argument("--bars", type=int, default=None, help="stop after this many bars")
    p.add_argument("--state-dir", type=Path, default=Path(".trader/paper"))
    p.add_argument("--force", action="store_true", help="run a rejected strategy anyway")
    p.set_defaults(func=cmd_paper)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
