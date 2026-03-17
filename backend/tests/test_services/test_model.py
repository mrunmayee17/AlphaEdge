"""Test PatchTST model components (legacy — kept for architecture validation)."""

import os

import torch
from alpha_model.model import AlphaModel, quantile_loss


def test_forward_pass():
    model = AlphaModel()
    ts = torch.randn(2, 23, 250)
    sec = torch.zeros(2, 11)
    sec[:, 0] = 1
    cap = torch.zeros(2, 5)
    cap[:, 0] = 1
    preds = model(ts, sec, cap)
    assert preds.shape == (2, 4, 3)


def test_quantile_loss():
    model = AlphaModel()
    ts = torch.randn(4, 23, 250)
    sec = torch.zeros(4, 11)
    sec[:, 0] = 1
    cap = torch.zeros(4, 5)
    cap[:, 0] = 1
    preds = model(ts, sec, cap)
    targets = torch.randn(4, 4) * 0.01
    loss = quantile_loss(preds, targets)
    assert loss.item() > 0
    loss.backward()


def test_predict_single():
    model = AlphaModel()
    ts = torch.randn(1, 23, 250)
    sec = torch.zeros(1, 11)
    sec[:, 0] = 1
    cap = torch.zeros(1, 5)
    cap[:, 0] = 1
    result = model.predict(ts, sec, cap)
    assert "alpha_21d" in result
    assert "q10_21d" in result
    assert "q90_21d" in result
    assert "inference_latency_ms" in result


def test_save_load(tmp_path):
    model = AlphaModel()
    path = str(tmp_path / "test_model.pt")
    model.save(path, fold="test")
    loaded = AlphaModel.load(path)
    ts = torch.randn(1, 23, 250)
    sec = torch.zeros(1, 11)
    sec[:, 0] = 1
    cap = torch.zeros(1, 5)
    cap[:, 0] = 1
    model.eval()
    loaded.eval()
    with torch.no_grad():
        p1 = model(ts, sec, cap)
        p2 = loaded(ts, sec, cap)
    assert torch.allclose(p1, p2, atol=1e-6)


def test_param_count():
    model = AlphaModel()
    params = model.count_parameters()
    assert params > 1_000_000  # should be ~2M
    assert params < 10_000_000  # should be well under 10M


def test_attention_weights():
    model = AlphaModel()
    ts = torch.randn(1, 23, 250)
    attn = model.patch_tst.get_attention_weights(ts)
    assert attn.shape == (1, 23, 50)
    assert abs(attn[0, 0].sum().item() - 1.0) < 0.01
