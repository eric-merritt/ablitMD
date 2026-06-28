"""Shared run picker for the bake / recipe CLI scripts.

Lets the scripts offer a numbered menu of usable runs instead of forcing the
caller to paste a long run id. A run is "usable" when it has a non-empty
capture directory and a matching run json.
"""

from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"


def runs_with_directories() -> list[str]:
  """Run ids backed by a non-empty capture directory and a run json."""
  backed = []
  for entry in sorted(RUNS_DIR.iterdir()):
    if not entry.is_dir() or entry.name.startswith("."):
      continue
    has_captures = any(entry.iterdir())
    has_run_json = (RUNS_DIR / f"{entry.name}.json").exists()
    if has_captures and has_run_json:
      backed.append(entry.name)
  return backed


def prompt_for_run(run_ids: list[str]) -> str:
  """Print a numbered menu and return the chosen run id."""
  print("Which run would you like to use?")
  for menu_number, run_id in enumerate(run_ids, start=1):
    print(f"  {menu_number}) {run_id}")
  selection = input("Enter number: ").strip()
  return run_ids[int(selection) - 1]


def resolve_run_id(cli_value: str | None) -> str:
  """Use the provided run id, otherwise show the directory-backed picker."""
  if cli_value:
    return cli_value
  candidates = runs_with_directories()
  if not candidates:
    raise SystemExit("no runs with capture directories found")
  return prompt_for_run(candidates)
