import torch
import torch.nn.functional as F
from onnx import helper
import onnx

from si4onnx.layers import HardSigmoid
from si4onnx.nn import NN


def _make_layer(alpha=None, beta=None):
    attrs = {}
    if alpha is not None:
        attrs["alpha"] = alpha
    if beta is not None:
        attrs["beta"] = beta
    node = helper.make_node("HardSigmoid", inputs=["x"], outputs=["y"], **attrs)
    # inputs are unused in __init__, but a dummy tensor keeps the signature consistent
    return HardSigmoid([torch.tensor(0.0)], node)


def test_forward_default_attrs_matches_clip():
    layer = _make_layer()  # uses default alpha=0.2, beta=0.5
    x = torch.tensor([-10.0, -3.0, 0.0, 3.0, 10.0])
    expected = torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)
    out = layer.forward(x)
    torch.testing.assert_close(out, expected)


def test_forward_matches_torch_hardsigmoid():
    alpha = 1.0 / 6.0
    beta = 0.5
    layer = _make_layer(alpha=alpha, beta=beta)
    x = torch.linspace(-5, 5, steps=11)
    out = layer.forward(x)
    expected = F.hardsigmoid(x)  # PyTorch定義: clamp(x/6 + 0.5, 0, 1)
    torch.testing.assert_close(out, expected)


def _export_hardsigmoid_onnx(tmp_path, alpha=1.0 / 6.0, beta=0.5):
    class HS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act = torch.nn.Hardsigmoid()

        def forward(self, x):
            return self.act(x)

    model = HS()
    x = torch.randn(1, 1)
    onnx_path = tmp_path / "hs.onnx"
    torch.onnx.export(
        model,
        x,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    return onnx_path, model


def test_onnx_load_and_forward(tmp_path):
    onnx_path, torch_model = _export_hardsigmoid_onnx(tmp_path)
    onnx_model = onnx.load(onnx_path)
    si_model = NN(model=onnx_model)

    x = torch.randn(1, 1)
    torch_out = torch_model(x).double()
    si_out = si_model.forward(x)

    torch.testing.assert_close(si_out, torch_out, atol=1e-6, rtol=1e-6)


def test_forward_si_interval(tmp_path):
    onnx_path, _ = _export_hardsigmoid_onnx(tmp_path)
    onnx_model = onnx.load(onnx_path)
    si_model = NN(model=onnx_model)

    x = torch.tensor([[0.1, -1.0, 5.0]], dtype=torch.double)
    a = x.clone()
    b = torch.ones_like(x, dtype=torch.double)
    l = torch.tensor(-1e6, dtype=torch.double)
    u = torch.tensor(1e6, dtype=torch.double)
    z = torch.tensor(0.0, dtype=torch.double)

    out, out_a, out_b, out_l, out_u = si_model.forward_si(x, a, b, l, u, z)

    # forward_si should keep interval finite and l < u
    assert out_l < out_u
    # outputs should be in [0,1] for HardSigmoid
    assert torch.all(out >= 0) and torch.all(out <= 1)
    # gradients for saturated parts should be zero
    assert torch.all(out_b[(x <= -3) | (x >= 3)] == 0)


def test_forward_si_middle_interval_default():
    # default alpha=0.2, beta=0.5 => bounds at -2.5, 2.5
    layer = _make_layer()
    a = torch.tensor([0.0], dtype=torch.double)
    b = torch.tensor([1.0], dtype=torch.double)
    l = torch.tensor(-1e6, dtype=torch.double)
    u = torch.tensor(1e6, dtype=torch.double)
    z = torch.tensor(0.0, dtype=torch.double)

    out_a, out_b, out_l, out_u = layer.forward_si(a, b, l, u, z)
    torch.testing.assert_close(out_l, torch.tensor(-2.5, dtype=torch.double))
    torch.testing.assert_close(out_u, torch.tensor(2.5, dtype=torch.double))
    torch.testing.assert_close(out_a, torch.tensor([0.5], dtype=torch.double))
    torch.testing.assert_close(out_b, torch.tensor([0.2], dtype=torch.double))
    assert out_l < out_u


def test_forward_si_lower_saturation_updates_u():
    layer = _make_layer()  # alpha=0.2, beta=0.5
    a = torch.tensor([-5.0], dtype=torch.double)
    b = torch.tensor([1.0], dtype=torch.double)
    l = torch.tensor(-1e6, dtype=torch.double)
    u = torch.tensor(1e6, dtype=torch.double)
    z = torch.tensor(0.0, dtype=torch.double)

    out_a, out_b, out_l, out_u = layer.forward_si(a, b, l, u, z)
    # lower saturation => output 0, b_out=0, upper bound should be ~2.5
    torch.testing.assert_close(out_a, torch.tensor([0.0], dtype=torch.double))
    torch.testing.assert_close(out_b, torch.tensor([0.0], dtype=torch.double))
    torch.testing.assert_close(out_u, torch.tensor(2.5, dtype=torch.double))
    assert out_l < out_u


def test_forward_si_upper_saturation_updates_l():
    layer = _make_layer()  # alpha=0.2, beta=0.5
    a = torch.tensor([5.0], dtype=torch.double)
    b = torch.tensor([1.0], dtype=torch.double)
    l = torch.tensor(-1e6, dtype=torch.double)
    u = torch.tensor(1e6, dtype=torch.double)
    z = torch.tensor(0.0, dtype=torch.double)

    out_a, out_b, out_l, out_u = layer.forward_si(a, b, l, u, z)
    # upper saturation => output 1, b_out=0, lower bound should be ~-2.5
    torch.testing.assert_close(out_a, torch.tensor([1.0], dtype=torch.double))
    torch.testing.assert_close(out_b, torch.tensor([0.0], dtype=torch.double))
    torch.testing.assert_close(out_l, torch.tensor(-2.5, dtype=torch.double))
    assert out_l < out_u
