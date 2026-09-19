import numpy as np

from backend.inference.som import train_som, compute_som_directions, select_best_layer
from backend.inference.recipe import build_som_md_recipe, directions_for_layer


class TestTrainSom:
  def test_output_shape(self):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 16)).astype(np.float32)
    weights = train_som(data, grid_shape=(4, 4), n_iter=50)
    assert weights.shape == (16, 16)

  def test_weights_near_data(self):
    """SOM weights should be near the data manifold after training."""
    rng = np.random.default_rng(0)
    # Two well-separated clusters.
    cluster_a = rng.standard_normal((50, 8)) + np.array([5.0] * 8)
    cluster_b = rng.standard_normal((50, 8)) - np.array([5.0] * 8)
    data = np.vstack([cluster_a, cluster_b]).astype(np.float32)
    weights = train_som(data, grid_shape=(4, 4), n_iter=1000)
    # Every weight should be closer to one of the cluster centers than to origin.
    for w in weights:
      dist_a = np.linalg.norm(w - cluster_a.mean(axis=0))
      dist_b = np.linalg.norm(w - cluster_b.mean(axis=0))
      assert min(dist_a, dist_b) < 15.0


class TestComputeSomDirections:
  def test_returns_k_unit_vectors(self):
    rng = np.random.default_rng(0)
    harmful = rng.standard_normal((50, 32)).astype(np.float32)
    centroid = np.ones(32, dtype=np.float32) * 10.0
    dirs = compute_som_directions(harmful, centroid, grid_shape=(4, 4), k=5)
    assert len(dirs) == 5
    for d in dirs:
      assert d.shape == (32,)
      np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-4)

  def test_directions_point_toward_centroid(self):
    """Each direction should have positive dot product with (centroid - neuron)."""
    rng = np.random.default_rng(0)
    harmful = rng.standard_normal((50, 16)).astype(np.float32)
    centroid = np.zeros(16, dtype=np.float32)  # origin as centroid
    dirs = compute_som_directions(harmful, centroid, grid_shape=(4, 4), k=4)
    # Directions point from neurons toward centroid (origin), so they should
    # be roughly opposite to the neuron positions.
    for d in dirs:
      assert np.linalg.norm(d) > 0.9


class TestSelectBestLayer:
  def test_returns_valid_index(self):
    rng = np.random.default_rng(0)
    harmful = rng.standard_normal((30, 10, 16)).astype(np.float32)
    harmless = rng.standard_normal((30, 10, 16)).astype(np.float32)
    best = select_best_layer(harmful, harmless)
    assert 0 <= best < 10

  def test_finds_separated_layer(self):
    """If layer 5 has a clear harmful/harmless separation, it should win."""
    rng = np.random.default_rng(0)
    n_h, n_l, dim = 20, 8, 16
    harmful = rng.standard_normal((n_h, n_l, dim)).astype(np.float32)
    harmless = rng.standard_normal((n_h, n_l, dim)).astype(np.float32)
    # Make layer 5 strongly separated.
    harmful[:, 5, :] += 10.0
    best = select_best_layer(harmful, harmless)
    assert best == 5


class TestBuildSomMdRecipe:
  def test_recipe_structure(self, tmp_path):
    from backend.inference.tests._fixtures import make_mini_run

    meta = make_mini_run(tmp_path)
    recipe = build_som_md_recipe(
      meta["run"], meta["model_id"], meta["gen_mode"],
      k=4, grid_shape=(3, 3), factor=0.8,
      state_dir=meta["state_dir"],
    )
    assert recipe["method"] == "som_md"
    assert recipe["k"] == 4
    assert recipe["factor"] == 0.8
    assert recipe["grid_shape"] == [3, 3]
    assert "best_layer" in recipe
    assert "per_layer_directions" in recipe

    # Each layer should have k directions of the right dimension.
    for layer_str, dirs in recipe["per_layer_directions"].items():
      assert len(dirs) == 4
      for d in dirs:
        assert len(d) == 16  # HIDDEN from fixture

  def test_directions_for_layer_dispatch(self, tmp_path):
    from backend.inference.tests._fixtures import make_mini_run

    meta = make_mini_run(tmp_path)
    recipe = build_som_md_recipe(
      meta["run"], meta["model_id"], meta["gen_mode"],
      k=3, grid_shape=(2, 2), factor=0.5,
      state_dir=meta["state_dir"],
    )
    # Layer 0 (embedding) should return nothing.
    assert directions_for_layer(recipe, hidden_index=0) == []

    # A decoder layer should return k directions.
    result = directions_for_layer(recipe, hidden_index=5)
    assert len(result) == 3
    for vec, factor in result:
      assert factor == 0.5
      np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-4)
