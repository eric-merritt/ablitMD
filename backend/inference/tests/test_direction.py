import numpy as np
import pytest
from backend.inference.direction import compute_direction, compute_similarity, compute_run_directions


def make_states(n_prompts: int, n_layers: int, hidden: int, seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return rng.standard_normal((n_prompts, n_layers, hidden)).astype(np.float32)


class TestComputeDirection:
  def test_output_shape(self):
    refusal = make_states(5, 10, 64, seed=0)
    non_refusal = make_states(5, 10, 64, seed=1)
    direction = compute_direction(refusal, non_refusal)
    assert direction.shape == (10, 64)

  def test_direction_is_unit_normalized(self):
    refusal = make_states(5, 10, 64, seed=0)
    non_refusal = make_states(5, 10, 64, seed=1)
    direction = compute_direction(refusal, non_refusal)
    norms = np.linalg.norm(direction, axis=1)
    np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)

  def test_zero_diff_layer_returns_zeros(self):
    states = make_states(3, 4, 8, seed=0)
    direction = compute_direction(states, states)
    np.testing.assert_array_equal(direction, np.zeros((4, 8)))

  def test_single_prompt_each_side(self):
    refusal = make_states(1, 6, 32, seed=2)
    non_refusal = make_states(1, 6, 32, seed=3)
    direction = compute_direction(refusal, non_refusal)
    assert direction.shape == (6, 32)


class TestComputeSimilarity:
  def test_output_shape(self):
    hidden_state = make_states(1, 10, 64, seed=0)[0]
    direction = make_states(1, 10, 64, seed=1)[0]
    similarity = compute_similarity(hidden_state, direction)
    assert similarity.shape == (10,)

  def test_identical_vectors_similarity_is_one(self):
    vec = make_states(1, 5, 32, seed=0)[0]
    similarity = compute_similarity(vec, vec)
    np.testing.assert_allclose(similarity, np.ones(5), atol=1e-5)

  def test_opposite_vectors_similarity_is_minus_one(self):
    vec = make_states(1, 5, 32, seed=0)[0]
    similarity = compute_similarity(vec, -vec)
    np.testing.assert_allclose(similarity, -np.ones(5), atol=1e-5)

  def test_range_is_minus_one_to_one(self):
    hidden_state = make_states(1, 20, 128, seed=4)[0]
    direction = make_states(1, 20, 128, seed=5)[0]
    similarity = compute_similarity(hidden_state, direction)
    assert (similarity >= -1.0 - 1e-5).all()
    assert (similarity <= 1.0 + 1e-5).all()


class TestComputeRunDirections:
  def test_groups_by_classification(self):
    hidden_states = {
      'p1__model__non_thinking': make_states(1, 4, 16, seed=0)[0],
      'p2__model__non_thinking': make_states(1, 4, 16, seed=1)[0],
      'p3__model__non_thinking': make_states(1, 4, 16, seed=2)[0],
      'p4__model__non_thinking': make_states(1, 4, 16, seed=3)[0],
    }
    classifications = [
      { 'hidden_states_key': 'p1__model__non_thinking', 'refused': True },
      { 'hidden_states_key': 'p2__model__non_thinking', 'refused': False },
      { 'hidden_states_key': 'p3__model__non_thinking', 'refused': True },
      { 'hidden_states_key': 'p4__model__non_thinking', 'refused': False },
    ]
    result = compute_run_directions(hidden_states, classifications, category='test_cat')
    assert 'direction_per_layer' in result
    assert 'similarity_per_prompt' in result
    assert len(result['similarity_per_prompt']) == 4
    for key, sims in result['similarity_per_prompt'].items():
      assert len(sims) == 4

  def test_returns_none_when_insufficient_data(self):
    hidden_states = {
      'p1__model__non_thinking': make_states(1, 4, 16, seed=0)[0],
    }
    classifications = [
      { 'hidden_states_key': 'p1__model__non_thinking', 'refused': True },
    ]
    result = compute_run_directions(hidden_states, classifications, category='test_cat')
    assert result is None
