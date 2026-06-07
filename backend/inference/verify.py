"""Stage 5 helpers — refusal heuristic + projection-strength metric."""

import numpy as np


def projection_strength(
    hidden_state: np.ndarray, direction: np.ndarray, layer_range: tuple[int, int]
) -> float:
    """Mean |cosine| of the unit hidden state onto the unit direction over an inclusive
    layer range. hidden_state: (n_layers, dim); direction: (dim,)."""
    unit_direction = direction / (np.linalg.norm(direction) + 1e-8)
    start, end = layer_range
    values = []
    for layer in range(start, min(end + 1, hidden_state.shape[0])):
        state = hidden_state[layer]
        unit_state = state / (np.linalg.norm(state) + 1e-8)
        values.append(abs(float(unit_state @ unit_direction)))
    return float(np.mean(values)) if values else 0.0
