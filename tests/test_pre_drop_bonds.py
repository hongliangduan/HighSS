"""Tests for --pre_drop_bonds: the bond feature is zeroed in the pre-run only."""
import inspect
import sys

from highss import design_engine as de


def test_pre_drop_bonds_is_wired_and_on_by_default(monkeypatch):
    params = inspect.signature(de.boltz_hallucination).parameters
    assert params["pre_drop_bonds"].default is False
    from highss import cli
    base = ["highss", "--target_pdb", "t.pdb", "--target_chains", "A", "--binder_length", "18",
            "--disulfide_pairs", "2-17", "--checkpoint", "m.ckpt", "--ccd_path", "ccd.pkl"]
    monkeypatch.setattr(sys, "argv", base)
    args = cli.parse_arguments()
    assert args.pre_drop_bonds is True and cli.load_config(args)["pre_drop_bonds"] is True
    monkeypatch.setattr(sys, "argv", base + ["--pre_drop_bonds", "false"])
    args = cli.parse_arguments()
    cfg = cli.load_config(args)
    assert cfg["pre_drop_bonds"] is False and cfg["loop_drop_bonds"] is False
    assert "pre_drop_bonds" in inspect.signature(de.boltz_hallucination).parameters   # config reaches the engine


def test_pre_drop_bonds_source_wiring():
    src = open(de.__file__, encoding="utf-8").read()
    assert src.count("if loop_drop_bonds or (pre_drop_bonds and pre_run):") == 1
    assert src.count("\n        drop_token_bonds(batch)\n") == 1          # still a single call site
    assert src.count('print("[pre_drop_bonds] token_bonds is zeroed in the pre-run only') == 1
    assert src.count('print("[loop_drop_bonds] token_bonds is zeroed inside the design loop (diagnostic)")') == 1
    # the block comes after the loop_fix_cys print and before the design loop is defined
    assert src.index("[loop_fix_cys] prescribed residues") < src.index("if loop_drop_bonds or (pre_drop_bonds and pre_run):") < src.index("    def design(batch,")
