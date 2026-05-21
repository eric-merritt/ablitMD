"""Stage 1 — cross-category divergence analysis.

Writes data/runs/<run_id>.divergence.json. CPU only, no GPU / inference service.

Usage: uv run python scripts/compute_divergence.py <run_id>
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from backend.inference.divergence import analyze_run

RUNS_DIR = _root / "data" / "runs"


def main():
  if len(sys.argv) < 2:
    print("usage: compute_divergence.py <run_id>")
    sys.exit(1)

  run_id = sys.argv[1]
  run = json.loads((RUNS_DIR / f"{run_id}.json").read_text())
  state_dir = RUNS_DIR / run_id

  out: dict = {"computed_at": datetime.now(timezone.utc).isoformat()}
  for step in run["sequence"]:
    model_id, gen_mode = step["model"], step["mode"]
    out.setdefault(model_id, {})[gen_mode] = analyze_run(run, model_id, gen_mode, state_dir)

  out_path = RUNS_DIR / f"{run_id}.divergence.json"
  out_path.write_text(json.dumps(out, indent=2))
  print(f"[divergence] wrote {out_path.name}")


if __name__ == "__main__":
  main()
