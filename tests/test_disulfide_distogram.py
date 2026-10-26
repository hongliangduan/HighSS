"""Checks the distogram-based disulfide term (run in the HighSS environment):
    python -m pytest tests/test_disulfide_distogram.py -q
"""
import pytest

torch = pytest.importorskip("torch")
from highss.design_engine import add_disulfide_distogram_loss, get_mid_points  # noqa: E402


def _batch(n_bind=8, n_tgt=3):
    ent = torch.tensor([[0] * n_bind + [1] * n_tgt])
    return {"entity_id": ent}, n_bind + n_tgt


def _logits(n, mid, centre, sharp=2.5):
    """Peaked distogram logits for every token pair at distance `centre`."""
    bins = torch.arange(mid.numel())
    k = int(torch.argmin(torch.abs(mid - centre)))
    base = -sharp * torch.abs(bins - k).float()
    x = base.expand(1, n, n, mid.numel()).clone()
    return x


def test_loss_low_inside_window_high_outside_and_has_gradient():
    batch, n = _batch()
    probe = torch.zeros(1, n, n, 64)
    mid = get_mid_points(probe)
    bonds = "1_SG_6_SG,2_SG_5_SG"          # 1-based pairs, index_base=1

    inside = _logits(n, mid, 4.0).requires_grad_(True)
    loss_in, mass_in = add_disulfide_distogram_loss(inside, mid, batch, bonds, 'A', 1)
    assert loss_in.item() < 0.05 and min(mass_in) > 0.95

    outside = _logits(n, mid, 9.0).requires_grad_(True)
    loss_out, mass_out = add_disulfide_distogram_loss(outside, mid, batch, bonds, 'A', 1)
    assert loss_out.item() > 0.95 and max(mass_out) < 0.05

    loss_out.backward()
    g = outside.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum().item() > 0
    # only the prescribed pairs (binder tokens 0,5 and 1,4) receive gradient
    nz = (g.abs().sum(-1) > 0).nonzero()
    assert set(map(tuple, nz[:, 1:].tolist())) <= {(0, 5), (1, 4)}


def test_no_pairs_returns_zero():
    batch, n = _batch()
    mid = get_mid_points(torch.zeros(1, n, n, 64))
    x = torch.zeros(1, n, n, 64, requires_grad=True)
    loss, rep = add_disulfide_distogram_loss(x, mid, batch, "", 'A', 1)
    assert loss.item() == 0.0 and rep == []


def test_nll_form_keeps_gradient_when_window_mass_is_small():
    import math
    batch, n = _batch()
    mid = get_mid_points(torch.zeros(1, n, n, 64))
    bonds = "1_SG_6_SG,2_SG_5_SG"
    win = ((mid >= 3.3) & (mid <= 4.8)).float()
    w = float(win.sum())
    s = math.log(0.02 * (64 - w) / (0.98 * w))            # window mass about 0.02
    base = (win * s).expand(1, n, n, 64).clone()
    res = {}
    for form in ("prob", "nll"):
        x = base.clone().requires_grad_(True)
        loss, rep = add_disulfide_distogram_loss(x, mid, batch, bonds, 'A', 1, form=form)
        loss.backward()
        res[form] = (loss.item(), x.grad.norm().item(), rep[0])
    assert abs(res["prob"][2] - 0.02) < 2e-3
    assert res["nll"][0] == pytest.approx(-math.log(res["nll"][2]), rel=1e-3)
    assert res["nll"][1] > 10 * res["prob"][1]


def test_bad_form_raises():
    batch, n = _batch()
    mid = get_mid_points(torch.zeros(1, n, n, 64))
    with pytest.raises(ValueError):
        add_disulfide_distogram_loss(torch.zeros(1, n, n, 64), mid, batch, "1_SG_6_SG", 'A', 1, form="oops")
