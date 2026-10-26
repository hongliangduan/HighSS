"""Checks of the --debug_sds_profile_every diagnostic (run in the HighSS environment):
    python -m pytest tests/test_sds_profile.py -q
"""
import math

import pytest

torch = pytest.importorskip("torch")


def test_profile_schedule():
    de = pytest.importorskip("highss.design_engine")
    ts = (1.5, 3.0, 6.0)
    de._SDS_PROFILE_STATE['calls'] = 0
    got = [de._sds_profile_ts("1_SG_4_SG", False, False, 2, ts) for _ in range(5)]
    assert got == [ts, (), ts, (), ts]
    # never in the pre-run, never without bonds, never when off; none of these advance the counter
    de._SDS_PROFILE_STATE['calls'] = 0
    assert de._sds_profile_ts("1_SG_4_SG", True, False, 2, ts) == ()
    assert de._sds_profile_ts("1_SG_4_SG", False, True, 2, ts) == ()
    assert de._sds_profile_ts("", False, False, 2, ts) == ()
    assert de._sds_profile_ts("1_SG_4_SG", False, False, 0, ts) == ()
    assert de._SDS_PROFILE_STATE['calls'] == 0


def test_cli_parse_floats():
    cli = pytest.importorskip("highss.cli")
    assert cli._parse_floats("1.5,3, 6") == (1.5, 3.0, 6.0)
    for bad in ("", "a,b", "1,-2", "0"):
        with pytest.raises(SystemExit):
            cli._parse_floats(bad)


def _batch(n_tok=6):
    a2t = torch.zeros(1, 6 * n_tok, n_tok)
    for t in range(n_tok):
        a2t[0, 6 * t:6 * t + 6, t] = 1.0
    return {"entity_id": torch.zeros(1, n_tok, dtype=torch.long), "atom_to_token": a2t,
            "atom_pad_mask": torch.ones(1, 6 * n_tok)}


def _coords(d_sg, n_tok=6):
    c = torch.zeros(1, 6 * n_tok, 3)
    c[0, :, 0] = torch.arange(6 * n_tok).float()
    sg1, sg2 = 5, 6 * 3 + 5
    c[0, sg1] = torch.tensor([0.0, 0.0, 0.0])
    c[0, sg2] = torch.tensor([d_sg, 0.0, 0.0])
    c[0, sg1 - 1] = torch.tensor([-0.5, 1.7, 0.0])
    c[0, sg2 - 1] = torch.tensor([d_sg + 0.5, -1.7, 0.0])
    return c


def test_profile_report_on_a_toy_graph(capsys):
    de = pytest.importorskip("highss.design_engine")
    n_tok = 6
    batch = _batch(n_tok)
    logits = torch.randn(1, n_tok, 33, requires_grad=True)
    batch["res_type_logits"] = logits
    p = torch.softmax(logits, dim=-1)

    def est(t):
        """Denoised estimate whose second SG moves with the sequence, more strongly for larger t."""
        c = _coords(4.0, n_tok).clone()
        c[0, 6 * 3 + 5, 0] = c[0, 6 * 3 + 5, 0] - (p[0, 3, 10:14].sum() * 2.0) * (t / 6.0)
        return c

    geom = dict(binder_bonds="1_SG_4_SG", binder_chain="A", index_base=1, target_dist=2.05, hinge_cutoff=2.3,
                mode="bounded", lower_bound=0.0, upper_bound=2.8, center_weight=0.0, angle_target=103.0,
                angle_tolerance=25.0, dihedral_target_abs=90.0, dihedral_tolerance=45.0)
    k_coords = est(1.0)
    k_terms = de.add_disulfide_distance_loss(k_coords, batch, **geom)
    losses = {
        "con_loss": p[0, :, 3:6].sum() * 0.1,
        "disulfide_distogram_loss": p[0, 1, 8:12].sum(),
        "disulfide_loss": k_terms[0], "disulfide_worst_pair_loss": k_terms[1],
        "disulfide_angle_loss": k_terms[2], "disulfide_dihedral_loss": k_terms[3],
    }
    scales = {"con_loss": 1.0, "disulfide_loss": 0.01, "disulfide_worst_pair_loss": 0.5,
              "disulfide_angle_loss": 0.0005, "disulfide_dihedral_loss": 0.0002, "disulfide_distogram_loss": 1e-6}
    out = {"sample_atom_coords": _coords(4.0, n_tok), "sample_atom_coords_grad": k_coords,
           "sds_profile": [(1.5, est(1.5)), (6.0, est(6.0))]}
    rows = de.sds_profile_report(batch, losses, scales, out, geom, binder_chain="A", stage="temp")
    text = capsys.readouterr().out
    assert "[sds_profile] stage=temp" in text and "t=1.50" in text and "t=6.00" in text
    assert [r["t"] for r in rows] == [1.5, 6.0]
    assert all(r["ratio"] and r["ratio"] > 0 and r["loss"] > 0 for r in rows)
    assert rows[1]["ratio"] > 2.0 * rows[0]["ratio"]          # estimate responds ~4x more at t = 6
    for r in rows:
        for k in ("cos_other", "cos_nll", "cos_k"):
            assert r[k] is None or -1.0 - 1e-6 <= r[k] <= 1.0 + 1e-6
    # the gradient the profile reports is the masked one: rows of the target and the Cys column are zero
    assert logits.grad is None
    assert math.isclose(rows[0]["d_hat"][0], 4.0 - 0.0, abs_tol=2.5)
