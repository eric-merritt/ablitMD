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


def _make_hook(directions: list[tuple[torch.Tensor, float]]):
  def hook(_module, _inputs, output):
    if isinstance(output, tuple):
      return (ablate_hidden(output[0], directions),) + tuple(output[1:])
    return ablate_hidden(output, directions)
  return hook


def set_ablation(recipe: dict, model) -> None:
  """Register a forward hook on each decoder layer per the recipe."""
  clear_ablation()
  global _ablation_recipe
  _ablation_recipe = recipe

  layers = model.model.layers
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
  layers = model.model.layers

  with torch.no_grad():
    for decoder_idx in range(len(layers)):
      raw = directions_for_layer(recipe, hidden_index=decoder_idx + 1)
      if not raw:
        continue
      layer = layers[decoder_idx]
      for vector, factor in raw:
        direction = torch.tensor(vector, device=device, dtype=dtype)
        for projection in (layer.self_attn.o_proj, layer.mlp.down_proj):
          projection.weight.copy_(
            orthogonalize_weight(projection.weight.data, direction, float(factor))
          )

  model.save_pretrained(out_path)
  tokenizer.save_pretrained(out_path)
  return out_path
