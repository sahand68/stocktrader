# Security scan: jarrodwatts/jev-trader

- Repo: https://github.com/jarrodwatts/jev-trader
- Commit: `b587759e459ea049590102e54a0b07800864cdc3` (2026-09-16)
- Scanned: 2026-09-26

## Summary

No secrets are in the git history, and nothing reachable from outside can make the bot place a trade. The most serious problem is that anyone can crash the bot through its public `/events` feed; this was reproduced locally. The other findings are safety limits that fail silently or don't hold.

## Method

- Manual review of every file under `src/`, `scripts/` and `web/src/`.
- `bun audit` on both lockfiles (`bun.lock`, `web/bun.lock`).
- Search of all 12 commits, including deleted files, for private keys, API keys and tokens.
- Local reproduction of the `/events` memory exhaustion against the real `startServer`.

## Findings

### 1. Anyone can run the bot out of memory through `/events` (High for a live bot)

**Where:** `src/server.ts:12-31`

**Problem:** Every connection to `/events` is kept with no limit and no authentication, and CORS is open to any origin (`*`). A client that connects and never reads makes the server buffer every new event for it in memory.

**Reproduction:** The real `startServer` was run with 1,000 events of history and one new ~1.3 KB event every 300 ms. 2,000 connections were opened from one machine and never read from. Server RSS went from 42 MB to 702 MB in about 60 seconds and kept rising.

**Impact:** The dashboard server and the trading loop run in one process, so crashing the server kills the bot. Each new connection also pulls about 1 MB of history, which makes bandwidth exhaustion cheap.

**Fix:** Cap connections overall and per IP. Drop any client whose buffer is backed up (`c.desiredSize < 0`). Send only the last N events on connect. Or move the feed into a separate process from the trader.

### 2. Two settings fail open to live trading or no position limit (Medium)

- **Dry run:** `src/config.ts:13` only treats `DRY_RUN=true` as a dry run. `DRY_RUN=1`, `TRUE` or `yes` with `PRIVATE_KEY` set trades with real money.
- **Position cap:** If `MAX_POSITION_MON` is something like `1_000` or `1000 MON`, it parses to NaN. The check at `src/trader.ts:213` (`Math.abs(exposure) > NaN`) is then always false, so the cap is off with no warning.

**Fix:** Parse settings strictly and refuse to start on an invalid value.

### 3. `MAX_POSITION_MON` isn't a hard limit (Medium)

**Where:** `src/trader.ts`, `src/market.ts`

**Problem:** Open orders and position are tracked only in memory.

- A transaction marked `lost` after 10 blocks (`src/market.ts:165-168`) that lands later leaves an order the bot never cancels and doesn't count toward the cap.
- A restart resets the position to zero and leaves the previous run's orders on the book.

**What limits the damage:** The Kuru margin balance (600 MON + 20 USDC by default).

**Fix:** On startup, read open orders and balances from the chain and cancel stale orders. Keep checking receipts for transactions marked lost.

### 4. No spending stop (Medium)

**Problem:** Every block sends a transaction, and Monad charges the full gas limit even on revert: about 0.03 MON per block, roughly 360 MON an hour. Nothing stops the bot on a run of reverts, a total gas budget, or a maximum loss. A bad RPC node or an order that keeps reverting drains the wallet at full speed.

**Fix:** Stop after N consecutive reverts, and add hourly gas and P&L limits.

### 5. Unlimited USDC approval (Low)

**Where:** `src/market.ts:238`

**Problem:** It approves `MaxUint256` to the margin account, so a compromise of that contract exposes all USDC in the wallet.

**Fix:** Approve only the amount being deposited.

### 6. RPC calls have no timeout (Low, availability)

**Where:** `src/chain.ts:4`; `readBook` at `src/market.ts:113` passes no `timeoutMs`.

**Problem:** One hung request leaves the loop marked busy, and every later block is skipped as late. The Jev model call has no timeout either.

**Mitigation already in place:** Orders are post-only, so a fake or stale book mostly causes reverts and wasted gas rather than trades at a bad price.

### 7. Orders are public before they land (Low)

**Where:** `src/index.ts:21`

**Problem:** Each order's side, price and transaction hash goes out on the public feed as soon as it is sent. Other traders can step one tick ahead of it before it is included in a block.

### 8. Build and deploy hygiene (Low)

- `bunfig.toml` sets `minimumReleaseAge = 0`, turning off Bun's waiting period for new package releases and removing a defence against a hijacked package update.
- The Docker image runs as root on a floating `oven/bun:1.3` tag.
- The Next.js dashboard sets no CSP or other security headers.

## Dependencies

- **Web app:** `bun audit` reports nothing.
- **Backend:** 11 advisories (1 critical, 3 high, 1 moderate, 6 low). None can be triggered in this app:
  - `elliptic@6.5.4` (critical, GHSA-vjh7-7g9h-fjfh, private key extraction): only in the `ethers@5.7.1` bundled inside `@kuru-labs/kuru-sdk`. The bot signs with the top-level `ethers@5.8.0`, which uses the patched `elliptic@6.6.1`; the SDK is only used for reads. The attack also needs the attacker to control what gets signed.
  - `ws@7.4.6` (DoS advisories): also only in that bundled ethers. The bot uses Bun's built-in WebSocket.
  - Fix anyway so the audit is clean: add `"overrides": { "elliptic": "^6.6.1", "ws": "^7.5.10" }` to `package.json`.

## Checked and clean

- **Secrets:** `.env` is excluded from git and the Docker image; no keys in history. `scripts/dry-encode.ts` and `scripts/trace-rpc.ts` only generate throwaway random wallets.
- **Model input:** Only numbers from the chain go into the model state, so there is no prompt injection route.
- **Web app:** No `dangerouslySetInnerHTML`. Transaction links use a fixed `https://monadvision.com/tx/` prefix with `rel="noreferrer"`.
- **Server:** Only read-only GET routes; nothing on it can change trading.
