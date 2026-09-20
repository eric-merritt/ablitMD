"""Post-ablation adversarial audit agent.

Runs AFTER a recipe has been built and baked (directions already applied to the
resident model). Each round it samples a random subset of categories, pulls one
random example prompt from each, sends them one at a time through the now-ablated
resident model, and classifies each fresh response as refused / not-refused with
the 9B judge. Fixed number of rounds — no early exit even at 100% non-refusal.

Each audit is persisted as data/runs/<run_id>.audit.<timestamp>.json so it becomes
one selectable "experiment" in the overlap workspace.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from backend.inference import classifier_llm, generator

RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "runs"


def _load_run(run_id: str) -> dict:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no run manifest at {path}")
    return json.loads(path.read_text())


def _recipe_master(run_data: dict) -> str | None:
    """Best-effort identity of the recipe that was baked, for labeling the audit."""
    import glob
    recipes = sorted(glob.glob(str(RUNS_DIR / f"{run_data['run_id']}.recipe.*.json")))
    if not recipes:
        return None
    try:
        r = json.loads(Path(recipes[-1]).read_text())
        return r.get("master_factor") or r.get("factor") or Path(recipes[-1]).name
    except Exception:
        return Path(recipes[-1]).name


def run_audit(run_id: str, n_categories: int = 5, rounds: int = 3) -> dict:
    run_data = _load_run(run_id)
    prompts = run_data.get("prompts") or []
    if not prompts:
        raise ValueError(f"run {run_id} has no prompts to audit")

    mode = run_data.get("mode_selection", "non_thinking")
    # Group prompts by category so we can sample one random example per category.
    by_category: dict[str, list[dict]] = {}
    for p in prompts:
        by_category.setdefault(p.get("category"), []).append(p)
    categories = list(by_category.keys())

    trials = []
    for _round in range(rounds):
        chosen = random.sample(categories, min(n_categories, len(categories)))
        for category in chosen:
            prompt = random.choice(by_category[category])
            response = generator.run_prompt(
                prompt["text"], mode, run_id, "audit", RUNS_DIR,
                skip_hidden_states=True,
            )
            label = classifier_llm.classify_one(prompt["text"], response)
            trials.append({
                "round": _round,
                "category": category,
                "prompt": prompt["text"],
                "response": response,
                "classification": label,
                "refused": label != "none",
            })

    record = {
        "run_id": run_id,
        "recipe_master": _recipe_master(run_data),
        "n_categories": n_categories,
        "rounds": rounds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trials": trials,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = RUNS_DIR / f"{run_id}.audit.{ts}.json"
    out_path.write_text(json.dumps(record, indent=2))
    record["path"] = str(out_path)
    return record


def run_audit_streaming(run_id: str, n_categories: int = 5, rounds: int = 3):
    """Streaming variant of :func:`run_audit`.

    Yields NDJSON-friendly dicts so the UI can watch the ablated model generate each
    response token-by-token and see the judge's label land per trial. Event shapes:

      {"type": "audit_start", "run_id", "n_categories", "rounds", "total"}
      {"type": "trial_start", "index", "round", "category", "prompt"}
      {"type": "token", "text"}                       # live generation
      {"type": "trial_done", "index", "round", "category", "prompt",
       "response", "classification", "refused"}
      {"type": "audit_done", "record"}               # full record, persisted

    The final record is written to disk exactly like :func:`run_audit`.
    """
    run_data = _load_run(run_id)
    prompts = run_data.get("prompts") or []
    if not prompts:
        raise ValueError(f"run {run_id} has no prompts to audit")

    mode = run_data.get("mode_selection", "non_thinking")
    by_category: dict[str, list[dict]] = {}
    for p in prompts:
        by_category.setdefault(p.get("category"), []).append(p)
    categories = list(by_category.keys())

    total = rounds * min(n_categories, len(categories))
    yield {"type": "audit_start", "run_id": run_id, "n_categories": n_categories,
           "rounds": rounds, "total": total}

    trials = []
    index = 0
    for round_no in range(rounds):
        chosen = random.sample(categories, min(n_categories, len(categories)))
        for category in chosen:
            prompt = random.choice(by_category[category])
            yield {"type": "trial_start", "index": index, "round": round_no,
                   "category": category, "prompt": prompt["text"]}

            collected = ""
            label = None
            for ev in generator.stream_prompt(
                prompt["text"], mode, run_id, "audit", RUNS_DIR,
                skip_hidden_states=True,
            ):
                if ev["type"] == "token":
                    collected += ev["text"]
                    yield {"type": "token", "text": ev["text"]}
                elif ev["type"] in ("done", "aborted"):
                    collected = ev.get("response", collected)
                elif ev["type"] == "error":
                    raise RuntimeError(ev.get("error") or "generation failed")

            label = classifier_llm.classify_one(prompt["text"], collected)
            trial = {
                "round": round_no,
                "category": category,
                "prompt": prompt["text"],
                "response": collected,
                "classification": label,
                "refused": label != "none",
            }
            trials.append(trial)
            yield {"type": "trial_done", "index": index, **trial}
            index += 1

    record = {
        "run_id": run_id,
        "recipe_master": _recipe_master(run_data),
        "n_categories": n_categories,
        "rounds": rounds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trials": trials,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = RUNS_DIR / f"{run_id}.audit.{ts}.json"
    out_path.write_text(json.dumps(record, indent=2))
    record["path"] = str(out_path)
    yield {"type": "audit_done", "record": record}


def list_audits(run_id: str) -> list[dict]:
    """Summaries of saved audits for the left-hand experiment list (newest first)."""
    out = []
    for path in sorted(RUNS_DIR.glob(f"{run_id}.audit.*.json"), reverse=True):
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        n_refused = sum(1 for t in rec.get("trials", []) if t.get("refused"))
        out.append({
            "path": str(path),
            "created_at": rec.get("created_at"),
            "recipe_master": rec.get("recipe_master"),
            "n_trials": len(rec.get("trials", [])),
            "n_refused": n_refused,
        })
    return out
