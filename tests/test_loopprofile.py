"""Tests for --loop_profile: the profile feature of the binder tokens inside the design loop.

mean = the RC3 behaviour: every binder token gets the averaged composition of the binder (the default of boltz_hallucination()).
token / token_pre = each binder token gets its own residue (the per-token profile of the true features), in every design stage /
in the pre-run only; token is the default of the command line since v0.2.0.  The per-token profile is the one the featurizer
builds for a fixed sequence: the mean of the MSA rows at the token.
"""
import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import highss.design_engine as de
from highss import cli

ROOT = Path(__file__).resolve().parents[1]
BINDER = de.chain_to_number["A"]
TARGET = de.chain_to_number["B"]
N_B, N_T, CH = 6, 3, 33
STAGES = {
    "pre": {"temp": 1.0, "soft": 1.0, "hard": 0.0},
    "soft": {"temp": 1.0, "soft": 0.5, "hard": 0.0},
    "hard": {"temp": 0.01, "soft": 1.0, "hard": 1.0},
}
ENGINE_SRC = Path(de.__file__).read_text(encoding="utf-8")


def _update_sequence():
    """The real nested update_sequence() of boltz_hallucination, extracted from the source and run on toy tensors."""
    nodes = [n for n in ast.walk(ast.parse(ENGINE_SRC)) if isinstance(n, ast.FunctionDef) and n.name == "update_sequence"]
    assert len(nodes) == 1
    lines = ENGINE_SRC.splitlines(keepends=True)[nodes[0].lineno - 1:nodes[0].end_lineno]
    ns = {"torch": torch, "device": torch.device("cpu"), "chain_to_number": de.chain_to_number, "loop_fix": None,
          "apply_loop_fix": de.apply_loop_fix}
    exec(textwrap.dedent("".join(lines)), ns)
    return ns["update_sequence"]


def _toy(seed=0, binder_first=True, rows=1):
    """A batch as the featurizer leaves it: random profile everywhere, `rows` MSA rows (1 = msa_mode none), binder tokens of entity A."""
    g = torch.Generator().manual_seed(seed)
    n = N_B + N_T
    ents = [BINDER] * N_B + [TARGET] * N_T if binder_first else [TARGET] * N_T + [BINDER] * N_B
    entity = torch.tensor([ents])
    logits = (torch.randn(1, n, CH, generator=g) * 2).requires_grad_(True)
    batch = {"entity_id": entity, "res_type_logits": logits, "res_type": torch.zeros(1, n, CH),
             "msa": torch.softmax(torch.randn(1, rows, n, CH, generator=g), dim=-1),
             "profile": torch.softmax(torch.randn(1, n, CH, generator=g), dim=-1)}
    mask = (entity[0] == BINDER).float()[None, :, None].expand(1, n, CH).contiguous()
    return batch, mask


def _run(opt_extra, stage="soft", seed=0, binder_first=True, rows=1):
    batch, mask = _toy(seed, binder_first, rows)
    before = {k: v.clone() for k, v in batch.items() if k in ("profile", "msa")}
    opt = dict(STAGES[stage])
    opt.update(opt_extra)
    _update_sequence()(opt, batch, mask)
    return batch, before


def _binder(batch):
    return batch["entity_id"][0] == BINDER


# ------------------------------------------------------------------------------------------------ behaviour of update_sequence()
@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("opt_extra", [{}, {"token_profile": False}])
@pytest.mark.parametrize("stage", list(STAGES))
def test_default_keeps_the_averaged_binder_profile(opt_extra, stage, rows):
    batch, before = _run(opt_extra, stage, rows=rows)
    b = _binder(batch)
    mean = batch["msa"][0, 0][b].float().mean(dim=0)                       # the binder's composition after this epoch's write
    assert torch.allclose(batch["profile"][0][b], mean.expand(int(b.sum()), CH), atol=1e-6)
    assert torch.allclose(batch["profile"][0][b][0], batch["profile"][0][b][-1])        # one and the same row for every binder token
    assert torch.equal(batch["profile"][0][~b], before["profile"][0][~b])               # target rows untouched
    assert not batch["profile"].requires_grad and not batch["msa"].requires_grad


@pytest.mark.parametrize("stage", list(STAGES))
def test_token_profile_is_the_own_residue_of_each_binder_token(stage):
    batch, before = _run({"token_profile": True}, stage)
    b = _binder(batch)
    assert batch["msa"].shape[1] == 1                                                               # msa_mode none: one row
    assert torch.allclose(batch["profile"][0][b], batch["msa"][0, 0][b].float(), atol=1e-6)
    assert torch.allclose(batch["profile"][0][b], batch["res_type"][0][b].detach(), atol=1e-6)      # = what was written to res_type
    assert not torch.allclose(batch["profile"][0][b][0], batch["profile"][0][b][1])                  # the rows differ from token to token
    assert torch.equal(batch["profile"][0][~b], before["profile"][0][~b])                           # target rows untouched
    assert not batch["profile"].requires_grad and not batch["msa"].requires_grad


@pytest.mark.parametrize("stage", list(STAGES))
def test_token_profile_with_several_msa_rows_is_the_row_mean_of_the_featurizer(stage):
    batch, before = _run({"token_profile": True}, stage, rows=3)
    b = _binder(batch)
    expect = batch["msa"][0][:, b].float().mean(dim=0)                                              # profile = msa.mean(dim=0) per token
    assert torch.allclose(batch["profile"][0][b], expect, atol=1e-6)
    assert not torch.allclose(batch["profile"][0][b], batch["msa"][0, 0][b].float())                # not just row 0
    assert torch.equal(batch["profile"][0][~b], before["profile"][0][~b])


def test_token_profile_of_a_hard_sequence_is_its_one_hot():
    batch, _ = _run({"token_profile": True}, "hard")
    b = _binder(batch)
    prof = batch["profile"][0][b]
    assert torch.allclose(prof.sum(dim=-1), torch.ones(int(b.sum())), atol=1e-6)
    assert prof.argmax(dim=-1).tolist() == batch["res_type"][0][b].argmax(dim=-1).tolist()          # the featurizer's profile of a 1-row MSA
    assert float(prof.max(dim=-1).values.min()) > 0.999


def test_token_profile_with_the_binder_after_the_target():
    batch, before = _run({"token_profile": True}, "soft", binder_first=False)
    b = _binder(batch)
    assert not bool(b[:N_T].any()) and bool(b[N_T:].all())
    assert torch.allclose(batch["profile"][0][b], batch["msa"][0, 0][b].float(), atol=1e-6)
    assert torch.equal(batch["profile"][0][~b], before["profile"][0][~b])


def test_the_sequence_gradient_is_the_same_in_both_modes():
    """The profile is detached in both modes: the gradient reaches res_type_logits through res_type only."""
    grads = {}
    for name, extra in (("mean", {}), ("token", {"token_profile": True})):
        batch, _ = _run(extra, "soft")
        g = torch.Generator().manual_seed(5)
        w_res, w_prof = torch.randn(*batch["res_type"].shape, generator=g), torch.randn(*batch["profile"].shape, generator=g)
        ((batch["res_type"] * w_res).sum() + (batch["profile"] * w_prof).sum()).backward()
        grads[name] = batch["res_type_logits"].grad.clone()
    assert torch.allclose(grads["mean"], grads["token"])
    assert float(grads["mean"].abs().sum()) > 0


def test_every_other_entry_of_the_batch_is_identical_in_both_modes():
    a, _ = _run({}, "soft", seed=2)
    b, _ = _run({"token_profile": True}, "soft", seed=2)
    for key in ("res_type", "msa", "soft", "hard", "pseudo", "logits"):
        assert torch.equal(a[key], b[key]), key


# ------------------------------------------------------------------------------------------------ which stages use it
def _flag_expression():
    lines = [l.strip() for l in ENGINE_SRC.splitlines() if l.strip().startswith('opt["token_profile"] =')]
    assert len(lines) == 1
    return lines[0].split("=", 1)[1].strip()


@pytest.mark.parametrize("mode,pre_run,expect", [
    ("mean", True, False), ("mean", False, False),
    ("token_pre", True, True), ("token_pre", False, False),
    ("token", True, True), ("token", False, True),
])
def test_the_modes_by_stage(mode, pre_run, expect):
    assert eval(_flag_expression(), {}, {"loop_profile": mode, "pre_run": pre_run}) is expect


# ------------------------------------------------------------------------------------------------ wiring
def test_the_source_wiring():
    src = ENGINE_SRC
    assert src.count('opt["token_profile"] =') == 1 and src.count("opt.get(\"token_profile\", False)") == 1
    assert src.index("    def design(batch,") < src.index('opt["token_profile"] =') < src.index("batch = update_sequence(opt, batch, mask")
    assert src.index("def update_sequence(") < src.index("opt.get(\"token_profile\", False)") < src.index("m = {\"soft\":[soft,e_soft]")
    assert src.count("if loop_profile not in ('mean', 'token_pre', 'token'):") == 1
    assert src.index("if loop_profile not in ('mean', 'token_pre', 'token'):") < src.index("predict_args = {")
    assert src.count("[loop_profile] {loop_profile}:") == 1
    assert src.count("    if loop_profile != 'mean':\n        print(f\"[loop_profile] {loop_profile}:") == 1      # printed in the token modes only
    assert src.index("[pre_drop_bonds] token_bonds is zeroed") < src.index("[loop_profile] {loop_profile}:") < src.index("    def design(batch,")
    # the default branch is the original expression, unchanged (row 0, mean over the binder tokens)
    assert src.count("_profile = batch['msa'][:, 0, _binder_sel, :].float().mean(dim=1)") == 1
    assert src.count("_profile = batch['msa'][:, :, _binder_sel, :].float().mean(dim=1).reshape(-1, batch['msa'].shape[-1])") == 1


def test_the_option_is_wired_and_token_is_the_command_line_default(monkeypatch):
    # the engine function keeps the RC3 behaviour as its own default; the command line (v0.2.0) uses the per-token profile
    assert inspect.signature(de.boltz_hallucination).parameters["loop_profile"].default == "mean"
    base = ["highss", "--target_pdb", "t.pdb", "--target_chains", "A", "--binder_length", "18",
            "--disulfide_pairs", "2-17", "--checkpoint", "m.ckpt", "--ccd_path", "ccd.pkl"]
    monkeypatch.setattr(sys, "argv", base)
    args = cli.parse_arguments()
    assert args.loop_profile == "token" and cli.load_config(args)["loop_profile"] == "token"
    for value in ("mean", "token_pre", "token"):
        monkeypatch.setattr(sys, "argv", base + ["--loop_profile", value])
        args = cli.parse_arguments()
        cfg = cli.load_config(args)
        assert args.loop_profile == value and cfg["loop_profile"] == value
        assert "loop_profile" in inspect.signature(de.boltz_hallucination).parameters         # the config reaches the engine
    monkeypatch.setattr(sys, "argv", base + ["--loop_profile", "token_post"])
    with pytest.raises(SystemExit):
        cli.parse_arguments()


def test_an_unknown_value_is_refused_before_anything_is_loaded():
    params = inspect.signature(de.boltz_hallucination).parameters
    kwargs = {name: None for name, p in params.items() if p.default is inspect.Parameter.empty}
    with pytest.raises(ValueError, match="loop_profile must be 'mean', 'token_pre' or 'token'"):
        de.boltz_hallucination(**kwargs, loop_profile="token_post")


def test_the_help_text_names_the_three_values():
    text = (ROOT / "highss" / "cli.py").read_text(encoding="utf-8")
    block = text[text.index('"--loop_profile"'):]
    block = block[:block.index("optimization.add_argument(")]
    for word in ("mean", "token_pre", "token", "pre-run only"):
        assert word in block
    assert re.search(r'choices=\["mean", "token_pre", "token"\], default="token"', block)
