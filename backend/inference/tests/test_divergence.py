import numpy as np

from backend.inference.divergence import (
  clumping_curve, mean_magnitude_curve, detect_onset, detect_divergence,
)


class TestClumpingCurve:
  def test_identical_directions_clump_to_one(self):
    # all categories point the same way at every layer -> clumping == 1
    d = np.zeros((3, 16), dtype=np.float32)
    d[:, 0] = 1.0
    curve = clumping_curve([d, d, d])
    np.testing.assert_allclose(curve, np.ones(3), atol=1e-5)

  def test_opposed_directions_clump_to_zero(self):
    # two categories pointing opposite ways -> centroid is zero -> clumping == 0
    layers = 4
    a = np.zeros((layers, 8), dtype=np.float32); a[:, 0] = 1.0
    b = np.zeros((layers, 8), dtype=np.float32); b[:, 0] = -1.0
    curve = clumping_curve([a, b])
    np.testing.assert_allclose(curve, np.zeros(layers), atol=1e-5)

  def test_output_length_matches_layers(self):
    d = np.ones((7, 8), dtype=np.float32)
    assert clumping_curve([d, d]).shape == (7,)


class TestMeanMagnitudeCurve:
  def test_averages_per_layer(self):
    a = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    b = np.array([3.0, 5.0, 7.0], dtype=np.float32)
    np.testing.assert_allclose(mean_magnitude_curve([a, b]), [2.0, 4.0, 6.0])


class TestDetectOnset:
  def test_first_layer_crossing_fraction_of_peak(self):
    mag = np.array([0.0, 0.1, 0.2, 1.0, 0.9], dtype=np.float32)
    # peak 1.0, frac 0.25 -> threshold 0.25 -> first index >= 0.25 is layer 3
    assert detect_onset(mag, frac=0.25) == 3

  def test_zero_magnitude_returns_zero(self):
    assert detect_onset(np.zeros(5, dtype=np.float32)) == 0


class TestDetectDivergence:
  def test_drop_below_retain_fraction(self):
    # peak after onset is 1.0; retain 0.9 -> first layer below 0.9
    clump = np.array([0.2, 0.95, 1.0, 0.98, 0.7, 0.5], dtype=np.float32)
    assert detect_divergence(clump, onset=1, retain=0.9) == 4

  def test_never_drops_returns_last_index(self):
    clump = np.array([0.5, 0.95, 1.0, 0.99], dtype=np.float32)
    assert detect_divergence(clump, onset=1, retain=0.9) == 3
