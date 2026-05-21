import numpy as np

from backend.inference.recipe import phase_a_direction, phase_b_direction, directions_for_layer


class TestPhaseADirection:
  def test_unit_normalized(self):
    rng = np.random.default_rng(0)
    dirs = [rng.standard_normal((10, 8)).astype(np.float32) for _ in range(3)]
    result = phase_a_direction(dirs, onset=2, split=6)
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-5)

  def test_averages_across_window_and_categories(self):
    a = np.zeros((10, 4), dtype=np.float32); a[:, 0] = 1.0
    b = np.zeros((10, 4), dtype=np.float32); b[:, 0] = 1.0
    result = phase_a_direction([a, b], onset=0, split=9)
    np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=1e-5)


class TestPhaseBDirection:
  def test_unit_normalized(self):
    rng = np.random.default_rng(1)
    dpl = rng.standard_normal((12, 8)).astype(np.float32)
    result = phase_b_direction(dpl, split=5, last=11)
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-5)


class TestDirectionsForLayer:
  def _recipe(self):
    return {
      "onset": 3, "split": 6, "last_layer": 9, "factor_a": 0.1, "factor_b": 0.2,
      "modes": {"hard": {
        "phase_a": {"layers": [3, 6], "direction": [1.0, 0.0]},
        "phase_b": {"layers": [6, 9], "per_category": {"cat_a": [0.0, 1.0], "cat_b": [1.0, 1.0]}},
      }},
    }

  def test_phase_a_layer_returns_mean_direction(self):
    result = directions_for_layer(self._recipe(), hidden_index=4)
    assert len(result) == 1
    vec, factor = result[0]
    assert factor == 0.1
    np.testing.assert_allclose(vec, [1.0, 0.0])

  def test_phase_b_layer_returns_all_categories(self):
    result = directions_for_layer(self._recipe(), hidden_index=8)
    assert len(result) == 2
    assert all(factor == 0.2 for _, factor in result)

  def test_layer_below_onset_returns_nothing(self):
    assert directions_for_layer(self._recipe(), hidden_index=1) == []
