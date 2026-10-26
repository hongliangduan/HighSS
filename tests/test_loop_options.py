"""Tests for --loop_drop_bonds, --seed and the sg_sg field of the epoch lines."""
import inspect
import random
import sys

import numpy as np
import torch

from highss import design_engine as de


def test_drop_token_bonds_zeroes_only_the_bond_feature():
    batch = {"token_bonds": torch.ones(1, 5, 5, 1), "res_type": torch.ones(1, 5, 33)}
    out = de.drop_token_bonds(batch)
    assert out is batch
    assert batch["token_bonds"].shape == (1, 5, 5, 1) and float(batch["token_bonds"].abs().sum()) == 0.0
    assert float(batch["res_type"].sum()) == 5 * 33


def test_format_sg_distances():
    assert de.format_sg_distances([2.3141, 2.05]) == "sg_sg:2.31/2.05"
    assert de.format_sg_distances([3.0]) == "sg_sg:3.00"


def test_seed_everything_is_reproducible():
    def draw(seed):
        de.seed_everything(seed)
        return random.random(), float(np.random.rand()), float(torch.rand(1))
    assert draw(5) == draw(5)
    assert draw(5) != draw(6)
    assert draw(2 ** 33 + 1)[0] == draw(2 ** 33 + 1)[0]   # large seeds are fine (numpy needs < 2**32)


def test_switches_are_wired_and_off_by_default(monkeypatch):
    params = inspect.signature(de.boltz_hallucination).parameters
    assert params["loop_drop_bonds"].default is False
    assert "seed" not in params                      # the seed is applied by the runner, not passed on
    from highss import cli
    base = ["highss", "--target_pdb", "t.pdb", "--target_chains", "A", "--binder_length", "18",
            "--disulfide_pairs", "2-17", "--checkpoint", "m.ckpt", "--ccd_path", "ccd.pkl"]
    monkeypatch.setattr(sys, "argv", base)
    args = cli.parse_arguments()
    cfg = cli.load_config(args)
    assert args.loop_drop_bonds is False and cfg["loop_drop_bonds"] is False
    assert args.seed is None and cfg["seed"] is None
    monkeypatch.setattr(sys, "argv", base + ["--loop_drop_bonds", "true", "--seed", "7"])
    args = cli.parse_arguments()
    cfg = cli.load_config(args)
    assert cfg["loop_drop_bonds"] is True and cfg["seed"] == 7


def test_every_config_key_except_seed_is_accepted_by_the_engine(monkeypatch):
    """boltz_hallucination(**filtered_config) must not receive an unknown keyword."""
    from highss import cli
    params = inspect.signature(de.boltz_hallucination).parameters
    monkeypatch.setattr(sys, "argv", ["highss", "--target_pdb", "t.pdb", "--target_chains", "A", "--binder_length", "18",
                                      "--disulfide_pairs", "2-17", "--checkpoint", "m.ckpt", "--ccd_path", "ccd.pkl",
                                      "--seed", "3", "--loop_drop_bonds", "true", "--loop_fix_cys", "true"])
    cfg = cli.load_config(cli.parse_arguments())
    src = open(de.__file__, encoding="utf-8").read()
    assert "'length_max', 'seed']" in src.replace('"', "'")          # seed is filtered out before the call
    for key in ("loop_drop_bonds", "loop_fix_cys", "set_train", "recycling_steps"):
        assert key in cfg and key in params


def test_engine_wiring_in_source():
    src = open(de.__file__, encoding="utf-8").read()
    assert src.count("\n        drop_token_bonds(batch)\n") == 1
    assert src.count("ss_distances = []") == 1
    assert src.count("loss_str.append(format_sg_distances(ss_distances))") == 1
    assert src.count("seed_everything(int(config['seed']) + itr)") == 1
    assert src.index("ss_distances = []") < src.index("loss_str.append(format_sg_distances(ss_distances))")
