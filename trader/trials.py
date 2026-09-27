"""
Every distinct backtest configuration run on a dataset is a trial, including the
ones you forget about. The ledger remembers them so the deflated Sharpe can't be
flattered by a short memory.
"""
import hashlib
import json
from pathlib import Path


class TrialLedger:
    def __init__(self, path: Path = Path(".trader/trials.jsonl")):
        self.path = path

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def record(self, dataset: dict, config: dict) -> int:
        """Log this configuration and return how many distinct ones were tried on this dataset."""
        dataset_key = json.dumps(dataset, sort_keys=True)
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        entries = self._entries()
        seen = {e["config_hash"] for e in entries if e["dataset"] == dataset_key}
        if config_hash not in seen:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"dataset": dataset_key, "config_hash": config_hash, "config": config}) + "\n")
            seen.add(config_hash)
        return len(seen)
