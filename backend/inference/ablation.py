"""Inference-time abliteration: in-place weight edit (verify) + weight bake (permanent)."""

from pathlib import Path

import numpy as np
import torch

from backend.inference.recipe import directions_for_layer


def ablate_hidden(hidden: torch.Tensor, directions: list[tuple[torch.Tensor, float]]) -> torch.Tensor:
  """Subtract each direction's projection from the hidden state.
  hidden: (..., dim); directions: list of (unit_direction (dim,), factor).
  Compute in float32 to avoid bf16 rounding silently zeroing the subtraction."""
  original_dtype = hidden.dtype
  work = hidden.to(torch.float32)
  for direction, factor in directions:
    dir_f32 = direction.to(torch.float32)
    coefficient = (work * dir_f32).sum(dim=-1, keepdim=True)
    work = work - factor * coefficient * dir_f32
  return work.to(original_dtype)


def orthogonalize_weight(weight: torch.Tensor, direction: torch.Tensor, factor: float) -> torch.Tensor:
  weight.addr_(direction, direction @ weight, alpha=-factor)
  return weight


def orthogonalize_input(weight: torch.Tensor, direction: torch.Tensor, factor: float) -> torch.Tensor:
  weight.addr_(weight @ direction, direction, alpha=-factor)
  return weight


def _decoder_layers(model):
  """Resolve the list of decoder layers, handling multimodal wrappers where
  the language tower lives under model.language_model rather than model directly."""
  inner = model.model
  if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
    return inner.language_model.layers
  return inner.layers


def _projection_modules(layer):
  projections = []
  if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
    projections.append(layer.self_attn.o_proj)
  if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
    projections.append(layer.linear_attn.out_proj)
  if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
    projections.append(layer.mlp.down_proj)
  return projections


# _mtp_projections is commented out: hidden_index=0 (the MTP head) maps to embed_tokens in the
# hidden-state array, and even factor=0.1 on layer 0 destroys output coherence entirely.
# Targeting the MTP head via o_proj has the same effect. Leave layer 0 alone.
# def _mtp_projections(model):
#   mtp = getattr(model, "mtp", None)
#   if mtp is None:
#     return []
#   projections = []
#   for layer in (mtp.layers if hasattr(mtp, "layers") else []):
#     if hasattr(layer, "self_attn"):
#       proj = getattr(layer.self_attn, "o_proj", None) or getattr(layer.self_attn, "out_proj", None)
#       if proj is not None:
#         projections.append(proj)
#   return projections


def apply_ablation_in_place(recipe: dict, model) -> dict:
  """Orthogonalize embedding + o_proj + down_proj in memory per the recipe.
  Returns a snapshot dict mapping each projection to its original weight clone
  so restore_model_weights can undo the edit without reloading from disk."""
  device = next(model.parameters()).device
  dtype = next(model.parameters()).dtype
  layers = _decoder_layers(model)
  snapshots: dict = {}

  _first_logged = False
  with torch.no_grad():
    # Layer 0 (embed_tokens / MTP head) is not ablated: even factor=0.1 destroys output entirely.
    # raw_emb = directions_for_layer(recipe, hidden_index=0)
    # if raw_emb:
    #   inner = model.model
    #   if hasattr(inner, "language_model"):
    #     inner = inner.language_model
    #   emb = inner.embed_tokens
    #   if id(emb) not in snapshots:
    #     snapshots[id(emb)] = (emb, emb.weight.data.clone())
    #   W = emb.weight.data.to(torch.float32)
    #   for vector, factor in raw_emb:
    #     d = torch.tensor(vector, device=device, dtype=torch.float32)
    #     proj_out = torch.outer(W @ d, d)
    #     W = W - float(factor) * proj_out
    #   emb.weight.copy_(W.to(dtype))

    for decoder_idx in range(len(layers)):
      raw = directions_for_layer(recipe, hidden_index=decoder_idx + 1)
      if not raw:
        continue
      for proj in _projection_modules(layers[decoder_idx]):
        original = proj.weight.data.clone()
        if id(proj) not in snapshots:
          snapshots[id(proj)] = (proj, original)
        # Work in float32 to avoid bf16 rounding zeroing the edit, cast back on copy.
        W = original.to(torch.float32)
        for vector, factor in raw:
          direction = torch.tensor(vector, device=device, dtype=torch.float32)
          W = orthogonalize_weight(W, direction, float(factor))
        proj.weight.copy_(W.to(dtype))
        if not _first_logged:
          _first_logged = True
          delta = float((proj.weight.data.to(torch.float32) - original.to(torch.float32)).norm())
          print(f"[ablation] first proj edit delta L2={delta:.6f} layer={decoder_idx} "
                f"proj={type(proj).__name__} shape={tuple(proj.weight.shape)}", flush=True)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
      all_directions = [
        (torch.tensor(v, device=device, dtype=torch.float32), float(f))
        for layer_dirs in (directions_for_layer(recipe, hidden_index=i + 1) for i in range(len(layers)))
        for v, f in layer_dirs
      ]
      if all_directions:
        if id(lm_head) not in snapshots:
          snapshots[id(lm_head)] = (lm_head, lm_head.weight.data.clone())
        W = lm_head.weight.data.to(torch.float32)
        for direction, factor in all_directions:
          W = orthogonalize_input(W, direction, factor)
        lm_head.weight.copy_(W.to(dtype))

    # MTP o_proj targeting removed — same reason as layer 0: destroys output coherence.
    # mtp_projs = _mtp_projections(model)
    # for mtp_proj in mtp_projs:
    #   if id(mtp_proj) not in snapshots:
    #     snapshots[id(mtp_proj)] = (mtp_proj, mtp_proj.weight.data.clone())
    #   W = mtp_proj.weight.data.to(torch.float32)
    #   for mode_data in recipe["modes"].values():
    #     d = torch.tensor(mode_data["phase_b"]["direction"], device=device, dtype=torch.float32)
    #     W = orthogonalize_weight(W, d, recipe["factor_b"])
    #   mtp_proj.weight.copy_(W.to(dtype))

    torch.cuda.synchronize(device)

  print(f"[ablation] apply_in_place: edited {len(snapshots)} projections across "
        f"onset={recipe['onset']} split={recipe['split']} last={recipe['last_layer']} "
        f"factor_a={recipe['factor_a']} factor_b={recipe['factor_b']} "
        f"modes={list(recipe['modes'].keys())}", flush=True)
  return snapshots


def restore_model_weights(snapshots: dict) -> None:
  """Restore all projection weights to the clones captured by apply_ablation_in_place."""
  with torch.no_grad():
    for proj, original in snapshots.values():
      proj.weight.copy_(original)


def compute_classic_directions(
  run_data: dict, state_dir: Path, model_id: str, gen_mode: str,
  include_disclaimer: bool = False, recipe: dict | None = None
) -> tuple[dict[int, np.ndarray], dict]:
  """Compute refusal directions + optionally phased disclaimer directions.

  Returns (refusal_dirs, disclaimer_result) where disclaimer_result is:
  - {} when include_disclaimer=False or no disclaimer states found
  - {"onset":int, "split":int, "phase_a":dict[int,list[ndarray]], "phase_b":dict[int,ndarray]}
    when recipe is provided (Phase A per-category onset..split, Phase B shared split+1..last)
  - dict[int,ndarray] flat when no recipe

  Post-ablation captures (disclaimer__{key}.npy) are merged if mean cosine vs originals > 0.5;
  otherwise discarded (direction has shifted under ablation).
  """
  refused_states:   list[np.ndarray]             = []
  complied_states:  list[np.ndarray]             = []
  orig_by_cat:      dict[str, list[np.ndarray]]  = {}
  capt_by_cat:      dict[str, list[np.ndarray]]  = {}

  for prompt in run_data["prompts"]:
    result = prompt.get("model_results", {}).get(model_id, {}).get(gen_mode)
    if not result:
      continue
    key      = result["hidden_states_key"]
    category = prompt["category"]
    npy_path = state_dir / f"{key}.npy"
    if not npy_path.exists():
      continue
    hidden       = np.load(str(npy_path))
    refusal_mode = result.get("refusal_mode", "none")

    if refusal_mode == "disclaimer":
      if include_disclaimer:
        orig_by_cat.setdefault(category, []).append(hidden)
    elif result.get("refused"):
      refused_states.append(hidden)
    else:
      complied_states.append(hidden)

    if include_disclaimer:
      disc_npy = state_dir / f"disclaimer__{key}.npy"
      if disc_npy.exists():
        capt_by_cat.setdefault(category, []).append(np.load(str(disc_npy)))

  if not refused_states or not complied_states:
    raise ValueError("need both refused and complied hidden states to compute classic directions")

  refused_arr  = np.stack(refused_states,  axis=0)
  complied_arr = np.stack(complied_states, axis=0)
  n_layers     = refused_arr.shape[1]

  directions: dict[int, np.ndarray] = {}
  for layer_idx in range(n_layers):
    diff = refused_arr[:, layer_idx, :].mean(axis=0) - complied_arr[:, layer_idx, :].mean(axis=0)
    norm = float(np.linalg.norm(diff))
    if norm > 1e-8:
      directions[layer_idx] = (diff / norm).astype(np.float32)

  if not include_disclaimer or (not orig_by_cat and not capt_by_cat):
    return directions, {}

  # Merge post-ablation captures if they're still pointing the same direction as originals.
  merged_by_cat: dict[str, list[np.ndarray]] = {cat: list(s) for cat, s in orig_by_cat.items()}
  if capt_by_cat:
    if orig_by_cat:
      all_orig = [h for s in orig_by_cat.values() for h in s]
      all_capt = [h for s in capt_by_cat.values() for h in s]
      orig_arr = np.stack(all_orig, axis=0)
      capt_arr = np.stack(all_capt, axis=0)
      cos_sims = []
      for layer_idx in range(n_layers):
        comp_mean = complied_arr[:, layer_idx, :].mean(0)
        o_diff = orig_arr[:, layer_idx, :].mean(0) - comp_mean
        c_diff = capt_arr[:, layer_idx, :].mean(0) - comp_mean
        o_norm = float(np.linalg.norm(o_diff))
        c_norm = float(np.linalg.norm(c_diff))
        if o_norm > 1e-8 and c_norm > 1e-8:
          cos_sims.append(float(np.dot(o_diff / o_norm, c_diff / c_norm)))
      mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
      if mean_cos > 0.5:
        print(f"[ablation] merging post-ablation disclaimer captures (cosine={mean_cos:.3f})", flush=True)
        for cat, states in capt_by_cat.items():
          merged_by_cat.setdefault(cat, []).extend(states)
      else:
        print(f"[ablation] discarding post-ablation captures — direction shifted (cosine={mean_cos:.3f})", flush=True)
    else:
      print("[ablation] using post-ablation disclaimer captures only (no originals)", flush=True)
      merged_by_cat = {cat: list(s) for cat, s in capt_by_cat.items()}

  if not merged_by_cat:
    return directions, {}

  all_disc    = [h for s in merged_by_cat.values() for h in s]
  disc_all_arr = np.stack(all_disc, axis=0)

  if recipe:
    onset = recipe["onset"]
    split = recipe["split"]
    last  = recipe["last_layer"]

    phase_a: dict[int, list[np.ndarray]] = {}
    for layer_idx in range(onset, min(split + 1, n_layers)):
      cat_dirs = []
      for cat, states in merged_by_cat.items():
        disc_arr = np.stack(states, axis=0)
        diff = disc_arr[:, layer_idx, :].mean(0) - complied_arr[:, layer_idx, :].mean(0)
        norm = float(np.linalg.norm(diff))
        if norm > 1e-8:
          cat_dirs.append((diff / norm).astype(np.float32))
      if cat_dirs:
        phase_a[layer_idx] = cat_dirs

    phase_b: dict[int, np.ndarray] = {}
    for layer_idx in range(split + 1, min(last + 1, n_layers)):
      diff = disc_all_arr[:, layer_idx, :].mean(0) - complied_arr[:, layer_idx, :].mean(0)
      norm = float(np.linalg.norm(diff))
      if norm > 1e-8:
        phase_b[layer_idx] = (diff / norm).astype(np.float32)

    return directions, {"onset": onset, "split": split, "phase_a": phase_a, "phase_b": phase_b}

  # Flat (no recipe)
  flat_disclaimer: dict[int, np.ndarray] = {}
  for layer_idx in range(n_layers):
    diff = disc_all_arr[:, layer_idx, :].mean(0) - complied_arr[:, layer_idx, :].mean(0)
    norm = float(np.linalg.norm(diff))
    if norm > 1e-8:
      flat_disclaimer[layer_idx] = (diff / norm).astype(np.float32)
  return directions, flat_disclaimer


def apply_classic_in_place(
  directions: dict[int, np.ndarray], factor: float, model,
  disclaimer: dict | None = None, disclaimer_factor: float = 0.3
) -> dict:
  """Apply classic (flat, per-layer) abliteration to model weights in memory.
  directions: {layer_index: unit_direction}. layer_index 0 = embedding, 1..N = decoder layers.

  disclaimer may be:
  - None / {} — no disclaimer ablation
  - {"onset":int, "split":int, "phase_a":dict[int,list[ndarray]], "phase_b":dict[int,ndarray]}
    for phased (recipe-aware) disclaimer
  - dict[int,ndarray] for flat disclaimer

  Returns weight snapshots for restore_model_weights.
  """
  device = next(model.parameters()).device
  dtype  = next(model.parameters()).dtype
  layers = _decoder_layers(model)
  snapshots: dict = {}

  def _snap_and_edit(proj, direction_t):
    if id(proj) not in snapshots:
      snapshots[id(proj)] = (proj, proj.weight.data.clone())
    W = proj.weight.data.to(torch.float32)
    proj.weight.copy_(orthogonalize_weight(W, direction_t.to(torch.float32), factor).to(dtype))

  # Unpack disclaimer into flat / phase_a / phase_b
  is_phased = isinstance(disclaimer, dict) and "onset" in disclaimer
  if is_phased:
    raw_phase_a: dict[int, list[np.ndarray]] = disclaimer.get("phase_a", {})
    raw_phase_b: dict[int, np.ndarray]       = disclaimer.get("phase_b", {})
    flat_dirs:   dict[int, np.ndarray]       = {}
  elif disclaimer:
    flat_dirs   = disclaimer  # type: ignore[assignment]
    raw_phase_a = {}
    raw_phase_b = {}
  else:
    flat_dirs   = {}
    raw_phase_a = {}
    raw_phase_b = {}

  # Gram-Schmidt: project out refusal component from each disclaimer direction.
  def _gs(d_disc: np.ndarray, layer_idx: int) -> np.ndarray | None:
    d_ref = directions.get(layer_idx)
    if d_ref is not None:
      d_res  = d_disc - float(np.dot(d_disc, d_ref)) * d_ref
      res_norm = float(np.linalg.norm(d_res))
      return (d_res / res_norm).astype(np.float32) if res_norm > 1e-4 else None
    return d_disc.copy()

  eff_flat:    dict[int, np.ndarray]       = {}
  eff_phase_a: dict[int, list[np.ndarray]] = {}
  eff_phase_b: dict[int, np.ndarray]       = {}

  for layer_idx, d in flat_dirs.items():
    r = _gs(d, layer_idx)
    if r is not None:
      eff_flat[layer_idx] = r

  for layer_idx, cat_dirs in raw_phase_a.items():
    eff = []
    for d in cat_dirs:
      r = _gs(d, layer_idx)
      if r is not None:
        eff.append(r)
    if eff:
      eff_phase_a[layer_idx] = eff

  for layer_idx, d in raw_phase_b.items():
    r = _gs(d, layer_idx)
    if r is not None:
      eff_phase_b[layer_idx] = r

  with torch.no_grad():
    # Refusal directions
    for layer_idx, direction in directions.items():
      direction_t = torch.tensor(direction, device=device, dtype=dtype)
      if layer_idx == 0:
        emb = model.model.embed_tokens
        if id(emb) not in snapshots:
          snapshots[id(emb)] = (emb, emb.weight.data.clone())
        emb.weight.data.addr_(direction_t @ emb.weight.data.T, direction_t, alpha=-factor)
        continue
      decoder_idx = layer_idx - 1
      if decoder_idx >= len(layers):
        continue
      for proj in _projection_modules(layers[decoder_idx]):
        _snap_and_edit(proj, direction_t)

    # Shared helpers for disclaimer edits (read current weight, not snapshot)
    def _edit_disc_proj(proj, direction_t):
      if id(proj) not in snapshots:
        snapshots[id(proj)] = (proj, proj.weight.data.clone())
      W = proj.weight.data.to(torch.float32)
      proj.weight.copy_(orthogonalize_weight(W, direction_t.to(torch.float32), disclaimer_factor).to(dtype))

    def _edit_disc_emb(emb, direction_t):
      if id(emb) not in snapshots:
        snapshots[id(emb)] = (emb, emb.weight.data.clone())
      emb.weight.data.addr_(
        direction_t.to(torch.float32) @ emb.weight.data.to(torch.float32).T,
        direction_t.to(torch.float32), alpha=-disclaimer_factor
      )

    def _apply_disc_at_layer(layer_idx: int, direction_t):
      if layer_idx == 0:
        _edit_disc_emb(model.model.embed_tokens, direction_t)
        return
      decoder_idx = layer_idx - 1
      if decoder_idx >= len(layers):
        return
      for proj in _projection_modules(layers[decoder_idx]):
        _edit_disc_proj(proj, direction_t)

    # Flat disclaimer
    for layer_idx, direction in eff_flat.items():
      _apply_disc_at_layer(layer_idx, torch.tensor(direction, device=device, dtype=dtype))

    # Phase A disclaimer (per-category directions, onset..split)
    for layer_idx, cat_dirs in eff_phase_a.items():
      for d in cat_dirs:
        _apply_disc_at_layer(layer_idx, torch.tensor(d, device=device, dtype=dtype))

    # Phase B disclaimer (shared direction, split+1..last)
    for layer_idx, direction in eff_phase_b.items():
      _apply_disc_at_layer(layer_idx, torch.tensor(direction, device=device, dtype=dtype))

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
      if id(lm_head) not in snapshots:
        snapshots[id(lm_head)] = (lm_head, lm_head.weight.data.clone())
      W = lm_head.weight.data.cpu().to(torch.float32)
      for direction in directions.values():
        W = orthogonalize_input(W, torch.tensor(direction, dtype=torch.float32), factor)
      for direction in eff_flat.values():
        W = orthogonalize_input(W, torch.tensor(direction, dtype=torch.float32), disclaimer_factor)
      for cat_dirs in eff_phase_a.values():
        for d in cat_dirs:
          W = orthogonalize_input(W, torch.tensor(d, dtype=torch.float32), disclaimer_factor)
      for direction in eff_phase_b.values():
        W = orthogonalize_input(W, torch.tensor(direction, dtype=torch.float32), disclaimer_factor)
      lm_head.weight.copy_(W.to(dtype).to(device))

    # MTP o_proj targeting removed — same reason as layer 0: destroys output coherence.
    # for mtp_proj in _mtp_projections(model):
    #   if id(mtp_proj) not in snapshots:
    #     snapshots[id(mtp_proj)] = (mtp_proj, mtp_proj.weight.data.clone())
    #   W = mtp_proj.weight.data.to(torch.float32)
    #   for direction in directions.values():
    #     W = orthogonalize_weight(W, torch.tensor(direction, dtype=torch.float32), factor)
    #   mtp_proj.weight.copy_(W.to(dtype).to(device))

  return snapshots


def compare_directions(
  recipe: dict, classic_directions: dict[int, np.ndarray]
) -> list[dict]:
  """Return per-layer comparison of ablitMD vs classic directions.
  Each entry: { layer, ablitmd_magnitude, classic_magnitude, cosine_similarity }.
  ablitMD magnitude is the mean norm of its direction vectors active at that layer;
  classic magnitude is the norm of the diff vector before unit-normalization (always 1.0
  post-norm, so we report the raw pre-norm magnitude separately if available — here we
  just report 1.0 as the classic direction is already unit-normalized)."""
  onset, split, last = recipe["onset"], recipe["split"], recipe["last_layer"]

  # Collect ablitMD direction vectors per layer from the recipe
  ablitmd_by_layer: dict[int, list[np.ndarray]] = {}
  for mode_data in recipe["modes"].values():
    for cat_dir in mode_data["phase_a"]["per_category"].values():
      phase_a_arr = np.array(cat_dir, dtype=np.float32)  # (n_slice_layers, dim)
      for layer_idx in range(onset, split + 1):
        offset = layer_idx - onset
        vec = phase_a_arr[offset] if phase_a_arr.ndim == 2 else phase_a_arr
        ablitmd_by_layer.setdefault(layer_idx, []).append(vec)
    phase_b_dir = np.array(mode_data["phase_b"]["direction"], dtype=np.float32)
    for layer_idx in range(split + 1, last + 1):
      ablitmd_by_layer.setdefault(layer_idx, []).append(phase_b_dir)

  all_layers = sorted(set(ablitmd_by_layer.keys()) | set(classic_directions.keys()))
  rows = []
  for layer_idx in all_layers:
    ablitmd_vecs = ablitmd_by_layer.get(layer_idx, [])
    classic_vec  = classic_directions.get(layer_idx)

    if ablitmd_vecs:
      ablitmd_mean = np.stack(ablitmd_vecs).mean(axis=0)
      norm = float(np.linalg.norm(ablitmd_mean))
      ablitmd_unit = ablitmd_mean / norm if norm > 1e-8 else ablitmd_mean
      ablitmd_magnitude = norm
    else:
      ablitmd_unit = None
      ablitmd_magnitude = None

    if classic_vec is not None:
      classic_norm = float(np.linalg.norm(classic_vec))
      classic_unit = classic_vec / classic_norm if classic_norm > 1e-8 else classic_vec
    else:
      classic_unit = None

    cosine = None
    if ablitmd_unit is not None and classic_unit is not None:
      cosine = float(np.clip(float(ablitmd_unit @ classic_unit), -1.0, 1.0))

    rows.append({
      "layer": layer_idx,
      "ablitmd_magnitude": ablitmd_magnitude,
      "classic_magnitude": 1.0 if classic_unit is not None else None,
      "cosine_similarity": cosine,
    })

  return rows


def directions_match_hook(baked: torch.Tensor, hooked: torch.Tensor, atol: float = 1e-5) -> bool:
  """True when a baked-weight output equals the equivalent hook output — the bake/hook
  equivalence check used by tests."""
  return bool(torch.allclose(baked, hooked, atol=atol))


def bake_and_save(recipe: dict, model, tokenizer, out_path: str) -> str:
  """Permanently orthogonalize o_proj + down_proj of each decoder layer against the
  recipe's directions, then save the modified model + tokenizer. Returns out_path."""
  apply_ablation_in_place(recipe, model)
  model.save_pretrained(out_path)
  tokenizer.save_pretrained(out_path)
  return out_path
