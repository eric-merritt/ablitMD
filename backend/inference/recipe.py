import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.inference.run_loader import rebuild_directions
from backend.inference.som import compute_som_directions, select_best_layer
import backend.inference.bo as bo


def _sidecar_has_vectors(data: dict) -> bool:
  """The directions sidecar / Mongo docs are normally stripped of direction_per_layer
  (the ~150MB per-layer vectors) since no chart reads them. Recipe building DOES need
  them, so only trust the sidecar when those vectors are actually present."""
  for cat_result in data.values():
    if not isinstance(cat_result, dict):
      continue
    by_mode = cat_result.get("by_mode")
    if not isinstance(by_mode, dict):
      continue
    for mode_data in by_mode.values():
      if isinstance(mode_data, dict) and mode_data.get("direction_per_layer"):
        return True
  return False


def load_directions(run: dict, model_id: str, gen_mode: str, state_dir: Path) -> dict:
  """Per-category directions for recipe building. Use the directions sidecar only when
  it carries the full per-layer vectors (a future full-directions pull); otherwise
  recompute from the raw .npy hidden states, which always have them."""
  sidecar = state_dir.parent / f"{run['run_id']}.directions.json"
  if sidecar.exists():
    data = json.loads(sidecar.read_text())
    if data and _sidecar_has_vectors(data):
      return {k: v for k, v in data.items() if isinstance(v, dict) and "by_mode" in v}
  return rebuild_directions(run, model_id, gen_mode, state_dir)


def recipe_filename(run_id: str) -> str:
  """Timestamped recipe filename: <run_id>.recipe.<YYYY-MM-DD_HHMMSS>.json.
  Every build writes a fresh file so each baked model is traceable to its recipe."""
  return f"{run_id}.recipe.{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"


def latest_recipe_path(runs_dir: Path, run_id: str):
  """Newest recipe for a run: most-recently-written of the timestamped files,
  falling back to a legacy <run_id>.recipe.json. Returns None if none exist."""
  candidates = list(runs_dir.glob(f"{run_id}.recipe.*.json"))
  legacy = runs_dir / f"{run_id}.recipe.json"
  if legacy.exists():
    candidates.append(legacy)
  if not candidates:
    return None
  return max(candidates, key=lambda path: path.stat().st_mtime)


def _renormalize(vector: np.ndarray) -> np.ndarray:
  norm = float(np.linalg.norm(vector))
  return vector / norm if norm > 1e-8 else vector


def shared_direction(directions: list[np.ndarray], start: int, end: int) -> np.ndarray:
  """Single unit vector: mean of every (category x layer) direction in [start..end],
  renormalized. directions: list of (n_layers, dim) per-category arrays."""
  stack = np.stack(directions, axis=0)            # (n_cat, n_layers, dim)
  window = stack[:, start:end + 1, :]             # (n_cat, w, dim)
  flat = window.reshape(-1, window.shape[-1])     # (n_cat*w, dim)
  return _renormalize(flat.mean(axis=0)).astype(np.float32)


def phase_a_direction(directions: list[np.ndarray], onset: int, split: int) -> np.ndarray:
  """Legacy wrapper for phase-A unit direction: shared_direction over the onset..split window."""
  return shared_direction(directions, onset, split)


def phase_b_direction(directions: np.ndarray, split: int, last: int) -> np.ndarray:
  """Legacy wrapper for phase-B unit direction: shared_direction over split..last."""
  return shared_direction([directions], split, last)


def dedup_overlap(pairs: list[tuple[np.ndarray, float]]) -> list[tuple[np.ndarray, float]]:
  """Ordered Gram-Schmidt so a subspace shared by several categories is ablated
  once, at the greatest factor among the categories that share it.

  Process directions highest-factor first; each lower-factor direction keeps only
  the component orthogonal to the directions already taken, ablated at its own
  factor. The shared part is therefore ablated solely by the highest-factor
  direction. pairs: [(unit_vector, factor), ...]; returns [(residual_unit, factor), ...]."""
  ordered = sorted(pairs, key=lambda pair: -pair[1])
  basis: list[np.ndarray] = []
  result: list[tuple[np.ndarray, float]] = []
  for vector, factor in ordered:
    residual = vector.astype(np.float32).copy()
    for prior in basis:
      residual = residual - float(residual @ prior) * prior
    norm = float(np.linalg.norm(residual))
    if norm <= 1e-6:
      continue  # fully covered by a higher-factor direction
    unit = (residual / norm).astype(np.float32)
    result.append((unit, factor))
    basis.append(unit)
  return result


def directions_for_layer(recipe: dict, hidden_index: int) -> list[tuple[np.ndarray, float]]:
  """Directions active at a hidden-state index.
  Dispatches to the SOM-MD handler for recipes with method="som_md",
  otherwise uses the two-phase (phase-A / phase-B) logic."""
  if recipe.get("method") == "som_md":
    return directions_for_layer_som_md(recipe, hidden_index)

  onset, split, last = recipe["onset"], recipe["split"], recipe["last_layer"]
  factor_a, factor_b = recipe["factor_a"], recipe["factor_b"]
  per_category_factor = recipe.get("factor_a_per_category") or {}
  result: list[tuple[np.ndarray, float]] = []

  for mode_data in recipe["modes"].values():
    if onset <= hidden_index <= split:
      # Legacy schema: mode_data["phase_a"] has "direction" directly
      if "per_category" in mode_data["phase_a"]:
        layer_offset = hidden_index - onset
        pairs: list[tuple[np.ndarray, float]] = []
        for category_id, per_layer in mode_data["phase_a"]["per_category"].items():
          arr = np.array(per_layer, dtype=np.float32)
          vec = arr[layer_offset] if arr.ndim == 2 else arr
          norm = float(np.linalg.norm(vec))
          if norm > 1e-8:
            factor = float(per_category_factor.get(category_id, factor_a))
            pairs.append((vec / norm, factor))
        result.extend(dedup_overlap(pairs))
      elif "direction" in mode_data["phase_a"]:
        vec = np.array(mode_data["phase_a"]["direction"], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
          result.append((vec / norm, factor_a))
    elif split < hidden_index <= last:
      # Phase B: legacy has per_category, new has direction
      if "per_category" in mode_data["phase_b"]:
        for cat_id, vec in mode_data["phase_b"]["per_category"].items():
          result.append((np.array(vec, dtype=np.float32), factor_b))
      elif "direction" in mode_data["phase_b"]:
        result.append((np.array(mode_data["phase_b"]["direction"], dtype=np.float32), factor_b))
  return result


def build_recipe(run: dict, model_id: str, gen_mode: str, onset: int, split: int,
                 factor_a: float, factor_b: float, state_dir,
                 factor_a_per_category: dict[str, float] | None = None,
                 last_layer: int | None = None) -> dict:
  """Build the two-phase abliteration recipe.
  Phase A (onset..split): one merged direction per category per layer
    (mean of hard + redirect unit vectors, renormalized).
  Phase B (split..last):  single shared direction (mean over all merged directions).
  last_layer: optional override to cap the final ablation layer (e.g. 35 instead of all)."""
  per_category = load_directions(run, model_id, gen_mode, state_dir)
  if not per_category:
    raise ValueError("no categories with directions")

  sample_mode = next(iter(next(iter(per_category.values()))["by_mode"].values()))
  total_layers = len(sample_mode["direction_per_layer"]) - 1
  last_layer = last_layer if last_layer is not None else total_layers
  # Clamp to valid range
  last_layer = min(last_layer, total_layers)

  merged_per_category: dict[str, np.ndarray] = {}
  for category_id, cat_result in per_category.items():
    by_mode = cat_result["by_mode"]
    hard  = by_mode.get("hard")
    redir = by_mode.get("redirect")

    if hard and redir:
      raw = (
        np.array(hard["direction_per_layer"],  dtype=np.float32) +
        np.array(redir["direction_per_layer"], dtype=np.float32)
      )
    elif hard:
      raw = np.array(hard["direction_per_layer"],  dtype=np.float32)
    elif redir:
      raw = np.array(redir["direction_per_layer"], dtype=np.float32)
    else:
      continue

    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    merged_per_category[category_id] = (raw / norms).astype(np.float32)

  if not merged_per_category:
    raise ValueError("no categories produced merged directions")

  shared = shared_direction(list(merged_per_category.values()), split, last_layer)

  return {
    "run_id": run["run_id"], "model_id": model_id, "gen_mode": gen_mode,
    "onset": onset, "split": split, "last_layer": last_layer,
    "factor_a": factor_a, "factor_b": factor_b,
    "factor_a_per_category": {
      cat_id: float(factor)
      for cat_id, factor in (factor_a_per_category or {}).items()
      if cat_id in merged_per_category
    },
    "modes": {
      "merged": {
        "phase_a": {
          "layers": [onset, split],
          "per_category": {
            cat_id: dpl[onset:split + 1, :].tolist()
            for cat_id, dpl in merged_per_category.items()
          },
        },
        "phase_b": {"layers": [split, last_layer], "direction": shared.tolist()},
      }
    },
    "built_at": datetime.now(timezone.utc).isoformat(),
  }


def build_som_md_recipe(
    run: dict,
    model_id: str,
    gen_mode: str,
    k: int = 44,
    grid_shape: tuple[int, int] = (7, 12),
    factor: float = 1.0,
    state_dir=None,
) -> dict:
    """Build a SOM-MD recipe: train a SOM on harmful hidden states at the best
    layer, compute k directions from top neurons toward the harmless centroid,
    and apply them uniformly across all decoder layers.

    Args:
        run: run dict with prompts + model_results.
        model_id: model identifier.
        gen_mode: generation mode key.
        k: number of SOM directions to use.
        grid_shape: SOM lattice dimensions (rows, cols).
        factor: ablation strength for all directions.
        state_dir: directory containing .npy hidden-state files.

    Returns:
        Recipe dict with "method": "som_md" and per-layer direction lists.
    """
    if state_dir is None:
        raise ValueError("state_dir required for SOM-MD recipe")

    # Load all hidden states, split into harmful (refused) vs harmless (complied).
    harmful_states: list[np.ndarray] = []
    harmless_states: list[np.ndarray] = []

    for prompt in run["prompts"]:
        result = prompt.get("model_results", {}).get(model_id, {}).get(gen_mode)
        if not result:
            continue
        key = result["hidden_states_key"]
        npy_path = state_dir / f"{key}.npy"
        if not npy_path.exists():
            continue
        hidden = np.load(str(npy_path))  # (n_layers, dim)
        if result.get("refused"):
            harmful_states.append(hidden)
        else:
            harmless_states.append(hidden)

    if not harmful_states or not harmless_states:
        raise ValueError(
            f"need both refused ({len(harmful_states)}) and complied "
            f"({len(harmless_states)}) hidden states for SOM-MD"
        )

    harmful_all = np.stack(harmful_states, axis=0)  # (n_h, n_layers, dim)
    harmless_all = np.stack(harmless_states, axis=0)  # (n_hl, n_layers, dim)

    # Select the best layer using the refusal metric.
    best_layer = select_best_layer(harmful_all, harmless_all, grid_shape=grid_shape)

    # Compute k SOM directions at that layer.
    harmful_at_layer = harmful_all[:, best_layer, :]  # (n_h, dim)
    harmless_centroid = harmless_all[:, best_layer, :].mean(axis=0)  # (dim,)

    directions = compute_som_directions(
        harmful_at_layer,
        harmless_centroid,
        grid_shape=grid_shape,
        k=k,
        hexagonal=True,
    )

    n_layers = harmful_all.shape[1]
    dim = harmful_all.shape[2]

    # Build per-layer direction lists: same k directions for every decoder layer.
    # Hidden-state index 0 is the embedding; decoder layers start at index 1.
    per_layer: dict[int, list[list[float]]] = {}
    for layer_idx in range(1, n_layers):
        per_layer[layer_idx] = [d.tolist() for d in directions]

    return {
        "run_id": run["run_id"],
        "model_id": model_id,
        "gen_mode": gen_mode,
        "method": "som_md",
        "k": k,
        "grid_shape": list(grid_shape),
        "best_layer": best_layer,
        "factor": factor,
        "n_layers": n_layers,
        "per_layer_directions": {str(idx): dirs for idx, dirs in per_layer.items()},
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


def directions_for_layer_som_md(recipe: dict, hidden_index: int) -> list[tuple[np.ndarray, float]]:
    """Directions active at a hidden-state index for a SOM-MD recipe.

    Returns the k SOM directions (each at `factor`) for any decoder layer.
    Hidden-state index 0 (embedding) returns nothing — same convention as
    the two-phase recipe."""
    if hidden_index <= 0:
        return []
    factor = float(recipe["factor"])
    dirs_raw = recipe.get("per_layer_directions", {}).get(str(hidden_index))
    if not dirs_raw:
        return []
    return [(np.array(d, dtype=np.float32), factor) for d in dirs_raw]


def _load_som_states(run: dict, model_id: str, gen_mode: str, state_dir):
    """Load hidden states and split into (harmful, harmless) stacks.

    Returns (harmful_all, harmless_all, val_prompts) where *_all are
    (n, n_layers, dim) arrays and val_prompts is the list of prompt dicts used
    for the held-out validation set (refused prompts only — those are the ones
    whose refusal we're trying to break)."""
    harmful_states: list[np.ndarray] = []
    harmless_states: list[np.ndarray] = []
    val_prompts: list[dict] = []

    for prompt in run["prompts"]:
        result = prompt.get("model_results", {}).get(model_id, {}).get(gen_mode)
        if not result:
            continue
        key = result["hidden_states_key"]
        npy_path = state_dir / f"{key}.npy"
        if not npy_path.exists():
            continue
        hidden = np.load(str(npy_path))  # (n_layers, dim)
        if result.get("refused"):
            harmful_states.append(hidden)
            val_prompts.append(prompt)
        else:
            harmless_states.append(hidden)

    return (
        np.stack(harmful_states, axis=0),
        np.stack(harmless_states, axis=0),
        val_prompts,
    )


def build_som_md_recipe_bo(
    run: dict,
    model_id: str,
    gen_mode: str,
    model,
    grid_shape: tuple[int, int] = (7, 12),
    n_trials: int = 50,
    factor_range: tuple[float, float] = (0.5, 1.5),
    state_dir=None,
    seed: int = 42,
    log_path=None,
) -> dict:
    """Closed-loop SOM-MD recipe: build a candidate direction pool from a hexagonal
    SOM, then run Bayesian Optimization over subsets of those directions + a global
    ablation factor, scoring each candidate by measured compliance rate on the
    held-out refused prompts. Returns the best-scoring recipe in the `som_md`
    schema (plus a "bo" block) so apply_ablation_in_place / bake work unchanged.

    Args:
        run: run dict with prompts + model_results.
        model_id: model identifier.
        gen_mode: generation mode key.
        model: the resident model (already loaded). Used by the scorer to apply
            ablations in place, generate, and restore.
        grid_shape: SOM lattice dimensions (rows, cols). Hexagonal by default.
        n_trials: BO budget.
        factor_range: (lo, hi) bounds for the global ablation factor.
        state_dir: directory containing .npy hidden-state files.
        seed: RNG seed for reproducibility.
        log_path: if given, append one JSON line per trial here.

    Returns:
        Recipe dict with "method": "som_md", the winning directions/factor, and a
        "bo" block describing the search.
    """
    from backend.inference.ablation import apply_ablation_in_place, restore_model_weights
    from backend.inference.generator import run_prompt
    from backend.inference.verify import auto_classify_response

    if state_dir is None:
        raise ValueError("state_dir required for SOM-MD recipe")

    harmful_all, harmless_all, val_prompts = _load_som_states(
        run, model_id, gen_mode, state_dir
    )
    if not len(harmful_all) or not len(harmless_all):
        raise ValueError(
            f"need both refused ({len(harmful_all)}) and complied "
            f"({len(harmless_all)}) hidden states for SOM-MD"
        )

    n_layers = harmful_all.shape[1]
    dim = harmful_all.shape[2]

    # Select the best layer using the refusal metric.
    best_layer = select_best_layer(harmful_all, harmless_all, grid_shape=grid_shape)

    # Full candidate pool: one direction per SOM neuron (no k cap).
    harmful_at_layer = harmful_all[:, best_layer, :]  # (n_h, dim)
    harmless_centroid = harmless_all[:, best_layer, :].mean(axis=0)  # (dim,)
    pool = compute_som_directions(
        harmful_at_layer,
        harmless_centroid,
        grid_shape=grid_shape,
        k=None,
        hexagonal=True,
    )

    # Deterministic validation split: hold out a small subset per category so the
    # scorer sees prompts the SOM never trained on. With only 10 prompts/category
    # we hold out at most 2; with more we hold out up to 10.
    from collections import defaultdict
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in val_prompts:
        cat = p.get("category") or p.get("id") or "unknown"
        by_cat[cat].append(p)
    holdout_per_cat = 2 if len(val_prompts) <= 440 else 10
    val_split: list[dict] = []
    for cat, prompts in by_cat.items():
        val_split.extend(prompts[:holdout_per_cat])

    run_id = run["run_id"]
    runs_dir = state_dir.parent

    def score_fn(indices: list[int], factor: float):
        """Apply the candidate ablation in place, generate on the validation set,
        classify each response, restore weights, and return (rate, n_ok, n_total)."""
        subset_dirs = [pool[i] for i in indices]
        per_layer = {str(idx): [d.tolist() for d in subset_dirs] for idx in range(1, n_layers)}
        trial_recipe = {
            "run_id": run_id,
            "model_id": model_id,
            "gen_mode": gen_mode,
            "method": "som_md",
            "k": len(subset_dirs),
            "grid_shape": list(grid_shape),
            "best_layer": best_layer,
            "factor": factor,
            "n_layers": n_layers,
            "per_layer_directions": per_layer,
        }
        snapshots = apply_ablation_in_place(trial_recipe, model)
        try:
            n_ok = 0
            for p in val_split:
                text = run_prompt(
                    p["text"], gen_mode, run_id, "bo_trial", runs_dir,
                    skip_hidden_states=True,
                )
                # Use the LLM judge, not the rule-based classifier.
                from backend.inference import classifier_llm
                if classifier_llm.classify_one(prompt_text=p["text"], response=text) == "none":
                    n_ok += 1
        finally:
            restore_model_weights(snapshots)
        rate = n_ok / len(val_split) if val_split else 0.0
        return rate, n_ok, len(val_split)

    result = bo.run_search(
        pool, score_fn,
        n_trials=n_trials,
        factor_range=factor_range,
        seed=seed,
        log_path=log_path,
    )

    if result.best is None:
        raise RuntimeError("BO search produced no trials")

    best = result.best
    subset_dirs = [pool[i] for i in best.indices]
    per_layer = {str(idx): [d.tolist() for d in subset_dirs] for idx in range(1, n_layers)}

    return {
        "run_id": run["run_id"],
        "model_id": model_id,
        "gen_mode": gen_mode,
        "method": "som_md",
        "k": len(subset_dirs),
        "grid_shape": list(grid_shape),
        "best_layer": best_layer,
        "factor": best.factor,
        "n_layers": n_layers,
        "per_layer_directions": per_layer,
        "bo": result.to_bo_block(),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
