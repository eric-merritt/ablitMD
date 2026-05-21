import torch

from backend.inference.ablation import ablate_hidden, orthogonalize_weight


class TestAblateHidden:
  def test_factor_one_removes_component(self):
    d = torch.tensor([1.0, 0.0, 0.0])
    h = torch.tensor([[3.0, 2.0, 0.0]])  # component 3 along d, 2 orthogonal
    out = ablate_hidden(h, [(d, 1.0)])
    assert abs(float(out[0, 0])) < 1e-6
    assert abs(float(out[0, 1]) - 2.0) < 1e-6

  def test_partial_factor_scales_component(self):
    d = torch.tensor([1.0, 0.0])
    h = torch.tensor([[5.0, 1.0]])
    out = ablate_hidden(h, [(d, 0.2)])
    assert abs(float(out[0, 0]) - 4.0) < 1e-6


class TestOrthogonalizeWeight:
  def test_output_component_reduced_by_factor(self):
    d = torch.tensor([1.0, 0.0, 0.0])
    weight = torch.tensor([[2.0, 1.0], [0.5, 3.0], [1.0, 1.0]])
    x = torch.tensor([1.0, 1.0])
    new_weight = orthogonalize_weight(weight, d, 1.0)
    out = new_weight @ x
    assert abs(float(out[0])) < 1e-5
