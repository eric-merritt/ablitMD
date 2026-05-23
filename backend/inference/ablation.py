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
  """Remove a direction from a weight matrix's output space.
  weight: (dim_out, dim_in) where dim_out == len(direction). Returns the edited matrix.
  Equivalent to ablate_hidden applied to every output the matrix produces."""
  return weight - factor * torch.outer(direction, direction @ weight)


def orthogonalize_input(weight: torch.Tensor, direction: torch.Tensor, factor: float) -> torch.Tensor:
  """Remove a direction from a weight matrix's input space.
  weight: (dim_out, dim_in) where dim_in == len(direction). Use for lm_head."""
  return weight - factor * torch.outer(weight @ direction, direction)


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
    # Ablate embedding layer (layer 0) - following abliterate_v2.py method
    raw_emb = directions_for_layer(recipe, hidden_index=0)
    if raw_emb:
      inner = model.model
      if hasattr(inner, "language_model"):
        inner = inner.language_model
      emb = inner.embed_tokens
      if id(emb) not in snapshots:
        snapshots[id(emb)] = (emb, emb.weight.data.clone())
      W = emb.weight.data.to(torch.float32)
      for vector, factor in raw_emb:
        d = torch.tensor(vector, device=device, dtype=torch.float32)
        proj_out = torch.outer(W @ d, d)
        W = W - float(factor) * proj_out
      emb.weight.copy_(W.to(dtype))

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
  run_data: dict, state_dir: Path, model_id: str, gen_mode: str, include_disclaimer: bool = False
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
  """Compute refusal + optionally disclaimer directions.
  Returns (refusal_directions, disclaimer_directions) dicts."""
  refused_states: list[np.ndarray] = []
  complied_states: list[np.ndarray] = []
  disclaimer_states: list[np.ndarray] = []

  for prompt in run_data["prompts"]:
    result = prompt.get("model_results", {}).get(model_id, {}).get(gen_mode)
    if not result:
      continue
    key = result["hidden_states_key"]
    npy_path = state_dir / f"{key}.npy"
    if not npy_path.exists():
      continue
    hidden = np.load(str(npy_path))
    refusal_mode = result.get("refusal_mode", "none")

    if refusal_mode == "disclaimer":
      if include_disclaimer:
        disclaimer_states.append(hidden)
    elif result.get("refused"):
      refused_states.append(hidden)
    else:
      complied_states.append(hidden)

  if not refused_states or not complied_states:
    raise ValueError("need both refused and complied hidden states to compute classic directions")

  refused_arr  = np.stack(refused_states,  axis=0)
  complied_arr = np.stack(complied_states, axis=0)

  n_layers = refused_arr.shape[1]
  directions: dict[int, np.ndarray] = {}
  for layer_idx in range(n_layers):
    diff = refused_arr[:, layer_idx, :].mean(axis=0) - complied_arr[:, layer_idx, :].mean(axis=0)
    norm = float(np.linalg.norm(diff))
    if norm < 1e-8:
      continue
    directions[layer_idx] = (diff / norm).astype(np.float32)

  disclaimer_directions: dict[int, np.ndarray] = {}
  if include_disclaimer and disclaimer_states:
    disc_arr = np.stack(disclaimer_states, axis=0)
    for layer_idx in range(n_layers):
      diff = disc_arr[:, layer_idx, :].mean(axis=0) - complied_arr[:, layer_idx, :].mean(axis=0)
      norm = float(np.linalg.norm(diff))
      if norm < 1e-8:
        continue
      disclaimer_directions[layer_idx] = (diff / norm).astype(np.float32)

  return directions, disclaimer_directions


def apply_classic_in_place(directions: dict[int, np.ndarray], factor: float, model,
                           disclaimer_directions: dict[int, np.ndarray] | None = None,
                           disclaimer_factor: float = 0.3) -> dict:
  """Apply classic (flat, per-layer) abliteration to model weights in memory.
  directions: {layer_index: unit_direction}. layer_index 0 = embedding, 1..N = decoder layers.
  Returns weight snapshots for restore_model_weights."""
  device = next(model.parameters()).device
  dtype = next(model.parameters()).dtype
  layers = _decoder_layers(model)
  snapshots: dict = {}

  def _snap_and_edit(proj, direction_t):
    if id(proj) not in snapshots:
      snapshots[id(proj)] = (proj, proj.weight.data.clone())
    W = proj.weight.data.to(torch.float32)
    proj.weight.copy_(orthogonalize_weight(W, direction_t.to(torch.float32), factor).to(dtype))

  with torch.no_grad():
    for layer_idx, direction in directions.items():
      direction_t = torch.tensor(direction, device=device, dtype=dtype)
      if layer_idx == 0:
        # embedding table: rows are output vectors written to the residual stream
        emb = model.model.embed_tokens
        if id(emb) not in snapshots:
          snapshots[id(emb)] = (emb, emb.weight.data.clone())
        W = emb.weight.data
        proj_out = direction_t.unsqueeze(1) * (direction_t @ W.T).unsqueeze(0)
        emb.weight.copy_(W - factor * proj_out.T)
        continue
      decoder_idx = layer_idx - 1
      if decoder_idx >= len(layers):
        continue
      for proj in _projection_modules(layers[decoder_idx]):
        _snap_and_edit(proj, direction_t)

    # Apply disclaimer directions with separate factor
    if disclaimer_directions:
      def _snap_and_edit_disc(proj, direction_t):
        if id(proj) not in snapshots:
          snapshots[id(proj)] = (proj, proj.weight.data.clone())
        W = proj.weight.data.to(torch.float32)
        proj.weight.copy_(orthogonalize_weight(W, direction_t.to(torch.float32), disclaimer_factor).to(dtype))

      for layer_idx, direction in disclaimer_directions.items():
        direction_t = torch.tensor(direction, device=device, dtype=dtype)
        if layer_idx == 0:
          emb = model.model.embed_tokens
          if id(emb) not in snapshots:
            snapshots[id(emb)] = (emb, emb.weight.data.clone())
          W = emb.weight.data
          proj_out = direction_t.unsqueeze(1) * (direction_t @ W.T).unsqueeze(0)
          emb.weight.copy_(W - disclaimer_factor * proj_out.T)
          continue
        decoder_idx = layer_idx - 1
        if decoder_idx >= len(layers):
          continue
        for proj in _projection_modules(layers[decoder_idx]):
          _snap_and_edit_disc(proj, direction_t)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
      if id(lm_head) not in snapshots:
        snapshots[id(lm_head)] = (lm_head, lm_head.weight.data.clone())
      W = lm_head.weight.data.to(torch.float32)
      for direction in directions.values():
        direction_t = torch.tensor(direction, device=device, dtype=torch.float32)
        W = orthogonalize_input(W, direction_t, factor)
      lm_head.weight.copy_(W.to(dtype))

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
    phase_a_dir = np.array(mode_data["phase_a"]["direction"], dtype=np.float32)
    for layer_idx in range(onset, split + 1):
      ablitmd_by_layer.setdefault(layer_idx, []).append(phase_a_dir)
    for cat_dir in mode_data["phase_b"]["per_category"].values():
      phase_b_arr = np.array(cat_dir, dtype=np.float32)  # (n_slice_layers, dim) or (dim,)
      for layer_idx in range(split + 1, last + 1):
        offset = layer_idx - split - 1
        vec = phase_b_arr[offset] if phase_b_arr.ndim == 2 else phase_b_arr
        ablitmd_by_layer.setdefault(layer_idx, []).append(vec)

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
