"""Inference-time abliteration: forward hooks (live) + weight bake (permanent)."""

import numpy as np
import torch

from backend.inference.recipe import directions_for_layer


def ablate_hidden(hidden: torch.Tensor, directions: list[tuple[torch.Tensor, float]]) -> torch.Tensor:
  """Subtract each direction's projection from the hidden state.
  hidden: (..., dim); directions: list of (unit_direction (dim,), factor)."""
  for direction, factor in directions:
    coefficient = (hidden * direction).sum(dim=-1, keepdim=True)
    hidden = hidden - factor * coefficient * direction
  return hidden


def orthogonalize_weight(weight: torch.Tensor, direction: torch.Tensor, factor: float) -> torch.Tensor:
  """Remove a direction from a weight matrix's output space.
  weight: (dim_out, dim_in) where dim_out == len(direction). Returns the edited matrix.
  Equivalent to ablate_hidden applied to every output the matrix produces."""
  return weight - factor * torch.outer(direction, direction @ weight)


_ablation_handles: list = []
_ablation_recipe: dict | None = None


_hook_fire_count = 0
_hook_diag_logged = False


def _make_hook(directions: list[tuple[torch.Tensor, float]]):
  def hook(_module, _inputs, output):
    global _hook_fire_count, _hook_diag_logged
    _hook_fire_count += 1
    is_tup = isinstance(output, tuple)
    if not _hook_diag_logged:
      _hook_diag_logged = True
      tensor = output[0] if is_tup else output
      print(f"[ablation] first hook fire: output_is_tuple={is_tup} "
            f"tensor_shape={tuple(tensor.shape) if hasattr(tensor, 'shape') else 'N/A'} "
            f"tensor_dtype={tensor.dtype if hasattr(tensor, 'dtype') else 'N/A'} "
            f"num_directions={len(directions)}", flush=True)
    if is_tup:
      return (ablate_hidden(output[0], directions),) + tuple(output[1:])
    return ablate_hidden(output, directions)
  return hook


def reset_hook_diag():
  global _hook_fire_count, _hook_diag_logged
  _hook_fire_count = 0
  _hook_diag_logged = False


def hook_fire_count():
  return _hook_fire_count


def _decoder_layers(model):
  """Resolve the list of decoder layers, handling multimodal wrappers where
  the language tower lives under model.language_model rather than model directly."""
  inner = model.model
  if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
    return inner.language_model.layers
  return inner.layers


def set_ablation(recipe: dict, model) -> None:
  """Register a forward hook on each decoder layer per the recipe."""
  clear_ablation()
  global _ablation_recipe
  _ablation_recipe = recipe

  layers = _decoder_layers(model)
  device = next(model.parameters()).device
  dtype = next(model.parameters()).dtype

  for decoder_idx in range(len(layers)):
    raw = directions_for_layer(recipe, hidden_index=decoder_idx + 1)
    if not raw:
      continue
    directions = [
      (torch.tensor(vector, device=device, dtype=dtype), float(factor))
      for vector, factor in raw
    ]
    _ablation_handles.append(layers[decoder_idx].register_forward_hook(_make_hook(directions)))


def clear_ablation() -> None:
  global _ablation_recipe
  for handle in _ablation_handles:
    handle.remove()
  _ablation_handles.clear()
  _ablation_recipe = None


def ablation_status() -> dict:
  return {
    "active": len(_ablation_handles) > 0,
    "run_id": _ablation_recipe["run_id"] if _ablation_recipe else None,
  }


def directions_match_hook(baked: torch.Tensor, hooked: torch.Tensor, atol: float = 1e-5) -> bool:
  """True when a baked-weight output equals the equivalent hook output — the bake/hook
  equivalence check used by tests."""
  return bool(torch.allclose(baked, hooked, atol=atol))


def bake_and_save(recipe: dict, model, tokenizer, out_path: str) -> str:
  """Permanently orthogonalize o_proj + down_proj of each decoder layer against the
  recipe's directions, then save the modified model + tokenizer. Returns out_path."""
  device = next(model.parameters()).device
  dtype = next(model.parameters()).dtype
  layers = _decoder_layers(model)

  with torch.no_grad():
    for decoder_idx in range(len(layers)):
      raw = directions_for_layer(recipe, hidden_index=decoder_idx + 1)
      if not raw:
        continue
      layer = layers[decoder_idx]
      projections = []
      if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        projections.append(layer.self_attn.o_proj)
      if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
        projections.append(layer.linear_attn.out_proj)
      if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        projections.append(layer.mlp.down_proj)

      for vector, factor in raw:
        direction = torch.tensor(vector, device=device, dtype=dtype)
        for projection in projections:
          projection.weight.copy_(
            orthogonalize_weight(projection.weight.data, direction, float(factor))
          )

  model.save_pretrained(out_path)
  tokenizer.save_pretrained(out_path)
  return out_path
