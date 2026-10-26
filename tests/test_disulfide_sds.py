"""Checks of the denoiser-estimate ("SDS-style") gradient path for the disulfide loss.

Run in the HighSS environment (needs torch + the vendored boltz package):
    python -m pytest tests/test_disulfide_sds.py -q
Toy denoiser, not the real model: the tests cover the sampler/loss plumbing only.
"""
import math

import pytest

torch = pytest.importorskip("torch")
from boltz.model.modules.diffusion import AtomDiffusion  # noqa: E402


class _Fake:
    denoise_grad = AtomDiffusion.denoise_grad

    def __init__(self):
        self.W = torch.randn(1, 1, 3, requires_grad=True)
        self.calls = []
        self.seen = []

    def preconditioned_network_forward(self, x, t, training, network_condition_kwargs):
        self.seen.append(x.detach().clone())
        self.calls.append((t, torch.is_grad_enabled(), network_condition_kwargs["model_cache"] is None,
                           network_condition_kwargs["multiplicity"], x.requires_grad))
        return 0.5 * x + self.W, None


def test_denoise_grad_graph_and_constant_structure():
    d = _Fake()
    x0 = torch.randn(1, 6, 3, requires_grad=True)
    outs = d.denoise_grad(x0, [1.5, 4.0], multiplicity=1, s_trunk=None)
    assert len(outs) == 2 and all(o.requires_grad for o in outs)
    assert [c[0] for c in d.calls] == [1.5, 4.0]
    # autograd on, conditioning recomputed (no cache), multiplicity passed on, structure detached
    assert all(c[1] and c[2] and c[3] == 1 and not c[4] for c in d.calls)
    sum(o.pow(2).sum() for o in outs).backward()
    assert d.W.grad is not None and torch.isfinite(d.W.grad).all() and d.W.grad.abs().sum() > 0
    assert x0.grad is None


def test_denoise_grad_noise_level_and_empty():
    d = _Fake()
    x0 = torch.zeros(1, 20000, 3)
    d.denoise_grad(x0, [2.0, 5.0], multiplicity=1)
    assert abs(d.seen[0].std().item() - 2.0) < 0.1 and abs(d.seen[1].std().item() - 5.0) < 0.25
    assert _Fake().denoise_grad(x0, [], multiplicity=1) == []


def test_sds_noise_levels_stratified_and_in_range():
    model = pytest.importorskip("boltz.model.model")
    f = model._sds_noise_levels
    for _ in range(200):
        ts = f(3, 1.5, 6.0)
        edges = [1.5 * (4.0 ** (k / 3)) for k in range(4)]
        assert len(ts) == 3 and all(edges[k] - 1e-9 <= ts[k] <= edges[k + 1] + 1e-9 for k in range(3))
    assert f(0, 1.5, 6.0) == [] and f(1, 3.0, 3.0) == [pytest.approx(3.0)]
    for bad in ((2, 0.0, 6.0), (2, 5.0, 2.0)):
        with pytest.raises(ValueError):
            f(*bad)


def _batch(n_tok=6):
    """Binder tokens 0..n_tok-1, each with 6 atoms (N, CA, C, O, CB, SG): SG = atom 6, CB = atom 5."""
    a2t = torch.zeros(1, 6 * n_tok, n_tok)
    for t in range(n_tok):
        a2t[0, 6 * t:6 * t + 6, t] = 1.0
    return {"entity_id": torch.zeros(1, n_tok, dtype=torch.long), "atom_to_token": a2t,
            "atom_pad_mask": torch.ones(1, 6 * n_tok)}


def _coords(d_sg, n_tok=6):
    """Cys 1 (tokens 0) and Cys 2 (token 3): SG atoms d_sg apart, CB atoms 1.8 A away from their SG."""
    c = torch.zeros(1, 6 * n_tok, 3)
    c[0, :, 0] = torch.arange(6 * n_tok).float()          # keep atoms apart
    sg1, sg2 = 5, 6 * 3 + 5
    c[0, sg1] = torch.tensor([0.0, 0.0, 0.0])
    c[0, sg2] = torch.tensor([d_sg, 0.0, 0.0])
    c[0, sg1 - 1] = torch.tensor([-0.5, 1.7, 0.0])
    c[0, sg2 - 1] = torch.tensor([d_sg + 0.5, -1.7, 0.0])
    return c


BONDS = "1_SG_4_SG"        # 1-based -> index_base=1


def test_distance_term_cap_and_gradient():
    de = pytest.importorskip("highss.design_engine")
    batch = _batch()
    kw = dict(binder_bonds=BONDS, binder_chain="A", index_base=1, mode="bounded", lower_bound=0.0, upper_bound=2.8)
    # mild violation: loss = (3.3 - 2.8)^2, gradient reaches the coordinates
    c = _coords(3.3).requires_grad_(True)
    loss, worst, _, _, rep = de.add_disulfide_distance_loss(c, batch, dist_term_cap=4.0, **kw)
    assert math.isclose(loss.item(), 0.25, rel_tol=1e-3) and math.isclose(worst.item(), 0.25, rel_tol=1e-3)
    loss.backward()
    assert c.grad.abs().sum() > 0
    # hopeless structure: value capped at clip^2, no gradient at all
    c = _coords(30.0).requires_grad_(True)
    loss, worst, _, _, _ = de.add_disulfide_distance_loss(c, batch, dist_term_cap=4.0, **kw)
    assert math.isclose(loss.item(), 4.0) and math.isclose(worst.item(), 4.0)
    loss.backward()
    assert c.grad.abs().sum() == 0
    # uncapped behaviour is unchanged
    loss, _, _, _, _ = de.add_disulfide_distance_loss(_coords(30.0), batch, **kw)
    assert math.isclose(loss.item(), (30.0 - 2.8) ** 2, rel_tol=1e-3)
    # inside the bound: zero and no gradient
    c = _coords(2.1).requires_grad_(True)
    loss, *_ = de.add_disulfide_distance_loss(c, batch, dist_term_cap=4.0, **kw)
    assert loss.item() == 0.0


def test_sds_losses_mean_over_levels_and_backward():
    de = pytest.importorskip("highss.design_engine")
    batch = _batch()
    geom = dict(binder_bonds=BONDS, binder_chain="A", index_base=1, mode="bounded", lower_bound=0.0,
                upper_bound=2.8, target_dist=2.05, hinge_cutoff=2.3, center_weight=0.0, angle_target=103.0,
                angle_tolerance=25.0, dihedral_target_abs=90.0, dihedral_tolerance=45.0)
    cs = [_coords(3.3).requires_grad_(True), _coords(30.0).requires_grad_(True)]
    loss, worst, ang, dih = de.disulfide_sds_losses(cs, batch, clip=2.0, **geom)
    assert math.isclose(loss.item(), (0.25 + 4.0) / 2, rel_tol=1e-3)
    assert math.isclose(worst.item(), (0.25 + 4.0) / 2, rel_tol=1e-3)
    (loss + worst + ang + dih).backward()
    assert cs[0].grad.abs().sum() > 0 and cs[1].grad.abs().sum() == 0
