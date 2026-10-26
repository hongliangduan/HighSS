"""Tests for --loop_fix_cys: the prescribed Cys positions are written into res_type inside the design loop."""
import ast
import inspect
import sys
import textwrap

import pytest
import torch

from highss import design_engine as de

ALPHABET = list('XXARNDCQEGHILKMFPSTWYV-')
N_B, N_T, CH = 6, 2, 33


def _update_sequence(loop_fix):
    """The real nested update_sequence() of boltz_hallucination, extracted from the source and run on toy tensors."""
    src = open(de.__file__, encoding="utf-8").read()
    nodes = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "update_sequence"]
    assert len(nodes) == 1
    lines = src.splitlines(keepends=True)[nodes[0].lineno - 1:nodes[0].end_lineno]
    ns = {"torch": torch, "device": torch.device("cpu"), "chain_to_number": {"A": 0, "B": 1},
          "loop_fix": loop_fix, "apply_loop_fix": de.apply_loop_fix}
    exec(textwrap.dedent("".join(lines)), ns)
    return ns["update_sequence"]


def _toy(seed=0):
    g = torch.Generator().manual_seed(seed)
    ent = torch.tensor([[0] * N_B + [1] * N_T])
    logits = (0.1 * torch.randn(1, N_B + N_T, CH, generator=g)).requires_grad_(True)
    batch = {"entity_id": ent, "res_type_logits": logits, "res_type": torch.zeros(1, N_B + N_T, CH),
             "msa": torch.zeros(1, 1, N_B + N_T, CH), "profile": torch.zeros(1, N_B + N_T, CH)}
    mask = torch.ones_like(logits)
    mask[ent != 0, :] = 0
    return batch, mask


STAGES = {"soft": {"temp": 1.0, "soft": 1.0, "hard": 0.0}, "hard": {"temp": 0.01, "soft": 1.0, "hard": 1.0}}


@pytest.mark.parametrize("stage", list(STAGES))
def test_original_loop_never_types_a_binder_residue_as_cys(stage):
    batch, mask = _toy()
    _update_sequence(None)(STAGES[stage], batch, mask)
    assert float(batch["res_type"][0, :N_B, 6].detach().abs().max()) < 1e-6          # Cys channel is excluded from the design alphabet
    assert 6 not in batch["res_type"][0, :N_B].argmax(dim=-1).tolist()


@pytest.mark.parametrize("stage", list(STAGES))
def test_loop_fix_writes_cys_into_res_type_msa_and_profile(stage):
    base, mask = _toy()
    _update_sequence(None)(STAGES[stage], base, mask)
    batch, mask = _toy()
    fix = de.build_loop_fix(batch["res_type_logits"], torch.arange(N_B), {1: "C", 4: "C"}, ALPHABET)
    _update_sequence(fix)(STAGES[stage], batch, mask)
    rt = batch["res_type"]
    for row in (1, 4):
        assert rt[0, row].argmax().item() == 6 and float(rt[0, row].sum()) == pytest.approx(1.0)
    others = [r for r in range(N_B) if r not in (1, 4)]
    assert torch.allclose(rt[0, others], base["res_type"][0, others])             # other positions untouched
    assert torch.allclose(rt[0, N_B:], base["res_type"][0, N_B:])                  # target untouched
    assert torch.allclose(batch["msa"][0, 0].float(), rt[0].detach())             # MSA row 0 follows res_type
    expect = rt[0, :N_B].detach().mean(dim=0)
    assert torch.allclose(batch["profile"][0, 0], expect)                          # the binder profile includes the Cys


def test_loop_fix_blocks_gradient_at_fixed_rows_only():
    batch, mask = _toy()
    fix = de.build_loop_fix(batch["res_type_logits"], torch.arange(N_B), {1: "C", 4: "C"}, ALPHABET)
    _update_sequence(fix)(STAGES["soft"], batch, mask)
    w = torch.randn_like(batch["res_type"])
    (batch["res_type"] * w).sum().backward()
    g = batch["res_type_logits"].grad
    assert float(g[0, [1, 4]].abs().max()) == 0.0
    assert float(g[0, [0, 2, 3, 5]].abs().max()) > 0.0


def test_fixed_residue_types():
    assert de.fixed_residue_types("2_C,6_C,13_C,17_C", 18, 1) == {1: "C", 5: "C", 12: "C", 16: "C"}
    assert de.fixed_residue_types("0_C,3_c", 5, 0) == {0: "C", 3: "C"}
    assert de.fixed_residue_types("", 5, 1) == {}
    with pytest.raises(ValueError):
        de.fixed_residue_types("20_C", 18, 1)
    with pytest.raises(ValueError):
        de.fixed_residue_types("2C", 18, 1)


def test_build_loop_fix():
    like = torch.zeros(1, 8, CH)
    one_hot, row_mask = de.build_loop_fix(like, torch.arange(6), {2: "C"}, ALPHABET)
    assert one_hot.shape == like.shape and one_hot[0, 2, 6] == 1.0 and one_hot.sum() == 1.0
    assert row_mask[0, 2].sum() == CH and row_mask.sum() == CH
    with pytest.raises(ValueError):
        de.build_loop_fix(like, torch.arange(6), {2: "B"}, ALPHABET)
    out = de.apply_loop_fix(torch.full_like(like, 0.5), one_hot, row_mask)
    assert out[0, 2, 6] == 1.0 and out[0, 2, 7] == 0.0 and out[0, 3, 6] == 0.5


def test_switch_is_wired_and_on_by_default(monkeypatch):
    # the engine function keeps its original default; the command line (v0.2.0) switches the fix on
    assert inspect.signature(de.boltz_hallucination).parameters["loop_fix_cys"].default is False
    from highss import cli
    base = ["highss", "--target_pdb", "t.pdb", "--target_chains", "A", "--binder_length", "18",
            "--disulfide_pairs", "2-17", "--checkpoint", "m.ckpt", "--ccd_path", "ccd.pkl"]
    monkeypatch.setattr(sys, "argv", base)
    args = cli.parse_arguments()
    assert args.loop_fix_cys is True and cli.load_config(args)["loop_fix_cys"] is True
    monkeypatch.setattr(sys, "argv", base + ["--loop_fix_cys", "false"])
    args = cli.parse_arguments()
    assert args.loop_fix_cys is False and cli.load_config(args)["loop_fix_cys"] is False
    cfg = cli.load_config(args)
    assert "loop_fix_cys" in inspect.signature(de.boltz_hallucination).parameters and cfg["loop_fix_cys"] is False
