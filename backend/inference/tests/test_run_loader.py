from backend.inference.run_loader import rebuild_directions
from backend.inference.tests._fixtures import make_mini_run, N_LAYERS, HIDDEN


class TestRebuildDirections:
  def test_returns_per_category_with_modes(self, tmp_path):
    meta = make_mini_run(tmp_path)
    result = rebuild_directions(meta["run"], meta["model_id"], meta["gen_mode"], meta["state_dir"])
    assert set(result.keys()) == {"cat_a", "cat_b"}
    assert set(result["cat_a"]["by_mode"].keys()) == {"hard", "redirect"}

  def test_direction_per_layer_shape(self, tmp_path):
    meta = make_mini_run(tmp_path)
    result = rebuild_directions(meta["run"], meta["model_id"], meta["gen_mode"], meta["state_dir"])
    hard = result["cat_a"]["by_mode"]["hard"]
    assert len(hard["direction_per_layer"]) == N_LAYERS
    assert len(hard["direction_per_layer"][0]) == HIDDEN
    assert len(hard["magnitude_per_layer"]) == N_LAYERS

  def test_skips_missing_npy(self, tmp_path):
    meta = make_mini_run(tmp_path)
    (meta["state_dir"] / "p2__Test__Model__non_thinking.npy").unlink()
    result = rebuild_directions(meta["run"], meta["model_id"], meta["gen_mode"], meta["state_dir"])
    assert result["cat_a"]["by_mode"]["hard"]["sample_count"] == 1
