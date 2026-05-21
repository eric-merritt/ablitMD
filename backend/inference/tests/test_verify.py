import numpy as np

from backend.inference.verify import looks_like_refusal, projection_strength


class TestLooksLikeRefusal:
  def test_detects_refusal_phrase(self):
    assert looks_like_refusal("I can't help with that request.")
    assert looks_like_refusal("I'm sorry, but I cannot assist.")

  def test_compliant_response_is_not_refusal(self):
    assert not looks_like_refusal("Sure! Here are the steps you asked for.")


class TestProjectionStrength:
  def test_aligned_state_has_strength_one(self):
    hidden = np.zeros((5, 4), dtype=np.float32)
    hidden[:, 0] = 1.0
    direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert abs(projection_strength(hidden, direction, (0, 4)) - 1.0) < 1e-5

  def test_orthogonal_state_has_strength_zero(self):
    hidden = np.zeros((5, 4), dtype=np.float32)
    hidden[:, 1] = 1.0
    direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert abs(projection_strength(hidden, direction, (0, 4))) < 1e-5
