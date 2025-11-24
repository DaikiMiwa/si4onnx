import torch
import torch.nn.functional as F
from onnx import helper
import onnx

from si4onnx.layers import HardTanh
from si4onnx.nn import NN


def _make_layer(min_val=None, max_val=None, op_type="HardTanh"):
    attrs = {}
    if min_val is not None:
        attrs["min"] = min_val
    if max_val is not None:
        attrs["max"] = max_val
    node = helper.make_node(op_type, inputs=["x"], outputs=["y"], **attrs)
    return HardTanh([torch.tensor(0.0)], node)


def test_forward_default_matches_clamp():
    layer = _make_layer()  # default min=-1, max=1
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = torch.clamp(x, min=-1.0, max=1.0)
    out = layer.forward(x)
    torch.testing.assert_close(out, expected)


def test_forward_custom_bounds_matches_hardtanh():
    min_val, max_val = -0.3, 0.8
    layer = _make_layer(min_val=min_val, max_val=max_val)
    x = torch.linspace(-1, 1, steps=7)
    out = layer.forward(x)
    expected = F.hardtanh(x, min_val=min_val, max_val=max_val)
    torch.testing.assert_close(out, expected)


def _export_hardtanh_onnx(tmp_path, min_val=-1.0, max_val=1.0):
    class HT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)

        def forward(self, x):
            return self.act(x)

    model = HT()
    x = torch.randn(1, 1)
    onnx_path = tmp_path / "ht.onnx"
    torch.onnx.export(
        model,
        x,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    return onnx_path, model


def test_onnx_clip_load_and_forward(tmp_path):
    onnx_path, torch_model = _export_hardtanh_onnx(tmp_path)
    onnx_model = onnx.load(onnx_path)
    si_model = NN(model=onnx_model)

    x = torch.randn(2, 3)
    torch_out = torch_model(x).double()
    si_out = si_model.forward(x)

    torch.testing.assert_close(si_out, torch_out, atol=1e-6, rtol=1e-6)


def test_forward_si_middle_interval_default():
    layer = _make_layer()
    a = torch.tensor([0.0], dtype=torch.double)
    b = torch.tensor([1.0], dtype=torch.double)
    l = torch.tensor(-1e6, dtype=torch.double)
    u = torch.tensor(1e6, dtype=torch.double)
    z = torch.tensor(0.0, dtype=torch.double)

    out_a, out_b, out_l, out_u = layer.forward_si(a, b, l, u, z)
    torch.testing.assert_close(out_l, torch.tensor(-1.0, dtype=torch.double))
    torch.testing.assert_close(out_u, torch.tensor(1.0, dtype=torch.double))
    torch.testing.assert_close(out_a, torch.tensor([0.0], dtype=torch.double))
    torch.testing.assert_close(out_b, torch.tensor([1.0], dtype=torch.double))
    assert out_l < out_u


def test_forward_si_lower_saturation_updates_u():
    layer = _make_layer()
    a = torch.tensor([-3.0], dtype=torch.double)
    b = torch.tensor([1.0], dtype=torch.double)
    l = torch.tensor(-1e6, dtype=torch.double)
    u = torch.tensor(1e6, dtype=torch.double)
    z = torch.tensor(0.0, dtype=torch.double)

    out_a, out_b, out_l, out_u = layer.forward_si(a, b, l, u, z)
    torch.testing.assert_close(out_a, torch.tensor([-1.0], dtype=torch.double))
    torch.testing.assert_close(out_b, torch.tensor([0.0], dtype=torch.double))
    torch.testing.assert_close(out_u, torch.tensor(2.0, dtype=torch.double))
    assert out_l < out_u


def test_forward_si_upper_saturation_updates_l():
    layer = _make_layer()
    a = torch.tensor([3.0], dtype=torch.double)
    b = torch.tensor([1.0], dtype=torch.double)
    l = torch.tensor(-1e6, dtype=torch.double)
    u = torch.tensor(1e6, dtype=torch.double)
    z = torch.tensor(0.0, dtype=torch.double)

    out_a, out_b, out_l, out_u = layer.forward_si(a, b, l, u, z)
    torch.testing.assert_close(out_a, torch.tensor([1.0], dtype=torch.double))
    torch.testing.assert_close(out_b, torch.tensor([0.0], dtype=torch.double))
    torch.testing.assert_close(out_l, torch.tensor(-2.0, dtype=torch.double))
    assert out_l < out_u


def test_forward_from_clip_inputs_uses_min_max_inputs():
    # Simulate Clip(min, max) inputs path
    node = helper.make_node("Clip", inputs=["x", "min", "max"], outputs=["y"])
    layer = HardTanh(
        [
            torch.tensor(0.0),
            torch.tensor(-0.5, dtype=torch.double),
            torch.tensor(0.5, dtype=torch.double),
        ],
        node,
    )
    x = torch.tensor([-1.0, -0.25, 0.25, 1.0], dtype=torch.double)
    expected = torch.clamp(x, min=-0.5, max=0.5)
    out = layer.forward(x)
    torch.testing.assert_close(out, expected)


def test_forward_from_clip_only_max():
    # Clip with only max (second input empty)
    node = helper.make_node("Clip", inputs=["x", "", "max"], outputs=["y"])
    layer = HardTanh(
        [
            torch.tensor(0.0),
            torch.tensor(0.5, dtype=torch.double),
        ],
        node,
    )
    x = torch.tensor([-1.0, 0.25, 1.0], dtype=torch.double)
    expected = torch.clamp(x, max=0.5)
    out = layer.forward(x)
    torch.testing.assert_close(out, expected)


def test_forward_from_clip_only_min():
    node = helper.make_node("Clip", inputs=["x", "min"], outputs=["y"])
    layer = HardTanh(
        [
            torch.tensor(0.0),
            torch.tensor(-0.5, dtype=torch.double),
        ],
        node,
    )
    x = torch.tensor([-1.0, 0.25, 1.0], dtype=torch.double)
    expected = torch.clamp(x, min=-0.5)
    out = layer.forward(x)
    torch.testing.assert_close(out, expected)
