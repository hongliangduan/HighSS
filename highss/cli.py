from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
import yaml

from .input_utils import (
    InputConfig,
    disulfide_cli_fields,
    generate_protein_binder_yaml,
    get_chain_residue_labels,
    get_chains_sequence,
    parse_disulfide_pairs,
    resolve_contact_residues,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"yes", "true", "t", "y", "1", "on"}:
        return True
    if lowered in {"no", "false", "f", "n", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# Optional input/loss preset selected with `--rc3_defaults true`
# (an option given explicitly on the command line always wins).
RC3_DEFAULTS = {
    "loop_fix_cys": False,
    "pre_drop_bonds": False,
    "loop_profile": "mean",
    "disulfide_distogram_loss": 0.0,
    "disulfide_distogram_form": "prob",
}


def _rc3_values_text() -> str:
    return ", ".join(
        f"--{key} {str(value).lower() if isinstance(value, bool) else value}" for key, value in RC3_DEFAULTS.items()
    )


# `--use_disulfide_loss false` removes every disulfide loss term: these options are set to 0, and a non-zero value given explicitly
# together with the switch is refused. The stage weights are in the table because, with --disulfide_schedule true, they and not
# --disulfide_loss decide the weight of the structure terms in each stage. The engine computes no term whose weight is 0.
LOSS_OFF = {
    "disulfide_loss": 0.0,
    "disulfide_loss_soft": 0.0,
    "disulfide_loss_temp": 0.0,
    "disulfide_loss_hard": 0.0,
    "disulfide_worst_pair_loss": 0.0,
    "disulfide_angle_loss": 0.0,
    "disulfide_dihedral_loss": 0.0,
    "disulfide_distogram_loss": 0.0,
    "disulfide_grad_steps": 0,
    "disulfide_sds_scale": 0.0,
    "debug_sds_profile_every": 0,
}


def _loss_off_text() -> str:
    return ", ".join(f"--{key}" for key in LOSS_OFF)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="highss",
        description=(
            "HighSS designs protein binders with a user-specified length and "
            "disulfide-bond topology."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_argument_group("protein target")
    target.add_argument("--target_pdb", required=True, help="Local target PDB file")
    target.add_argument(
        "--target_chains",
        required=True,
        help="Comma-separated target chains from the PDB, e.g. A or A,B",
    )
    target.add_argument("--target_name", default="", help="Run target name; defaults to PDB stem")
    target.add_argument("--binder_chain", default="A", help="Output binder chain ID")
    target.add_argument(
        "--contact_residues",
        default="",
        help="Optional comma-separated target residue numbers for pocket conditioning",
    )
    target.add_argument(
        "--contact_target_chain",
        default="",
        help="Original PDB chain receiving contact constraints; required for multi-chain targets",
    )
    target.add_argument(
        "--msa_mode",
        choices=["none", "target"],
        default="target",
        help="target=generate MSA for target chains only; none=disable MSA. The binder remains single-sequence.",
    )
    target.add_argument("--msa_max_seqs", type=int, default=4096)

    topology = parser.add_argument_group("binder topology")
    topology.add_argument("--binder_length", type=int, required=True, help="Exact binder length")
    topology.add_argument(
        "--disulfide_pairs",
        required=True,
        help="1-based Cys pair topology, e.g. 2-17,6-13",
    )

    paths = parser.add_argument_group("paths and execution")
    paths.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the HighSS model checkpoint",
    )
    paths.add_argument(
        "--ccd_path",
        required=True,
        help="Path to the local ccd.pkl dictionary",
    )
    paths.add_argument("--work_dir", default=".")
    paths.add_argument("--suffix", default="run")
    paths.add_argument("--gpu_id", type=int, default=0)
    paths.add_argument("--design_samples", type=int, default=1)
    paths.add_argument("--redo_boltz_predict", type=str2bool, default=True)
    paths.add_argument("--show_animation", type=str2bool, default=False)
    paths.add_argument("--save_trajectory", type=str2bool, default=False)

    optimization = parser.add_argument_group("design optimization")
    optimization.add_argument("--optimizer_type", choices=["SGD", "AdamW"], default="SGD")
    optimization.add_argument("--pre_iteration", type=int, default=30)
    optimization.add_argument("--soft_iteration", type=int, default=80)
    optimization.add_argument("--temp_iteration", type=int, default=45)
    optimization.add_argument("--hard_iteration", type=int, default=5)
    optimization.add_argument("--semi_greedy_steps", type=int, default=2)
    optimization.add_argument("--recycling_steps", type=int, default=1)
    optimization.add_argument(
        "--loop_fix_cys", type=str2bool, default=True,
        help="Inside the design loop, write the prescribed Cys (--disulfide_pairs) into res_type / MSA / profile, so that the "
             "model sees Cys at those positions in every forward pass. With false, those positions are "
             "typed as other amino acids in the loop.",
    )
    optimization.add_argument(
        "--loop_drop_bonds", type=str2bool, default=False,
        help="Diagnostic only: inside the design loop the model gets no bond feature (token_bonds zeroed). "
             "The semi-greedy and final predictions still use the bond constraints. Default false.",
    )
    optimization.add_argument(
        "--pre_drop_bonds", type=str2bool, default=True,
        help="No bond feature (token_bonds zeroed) in the pre-run only; the design stages, the semi-greedy step and the final "
             "prediction keep it. Without the bond feature the distogram Cys-Cys term (--disulfide_distogram_loss) has something "
             "to optimise in the pre-run, which sets the sequence that the later stages start from. false keeps the bond feature.",
    )
    optimization.add_argument(
        "--loop_profile", choices=["mean", "token_pre", "token"], default="token",
        help="MSA profile feature of the binder tokens inside the design loop. token: each binder token gets its own "
             "residue, as in the true features that boltz predict feeds the model (every design stage). token_pre: the same "
             "in the pre-run only. mean: every binder token gets the averaged composition of the binder.",
    )
    optimization.add_argument(
        "--seed", type=int, default=None,
        help="Seed python/numpy/torch at the start of every design (seed + design index). Default: unseeded.",
    )
    optimization.add_argument(
        "--set_train", type=str2bool, default=True,
        help="Design-loop forward pass in train mode (dropout on). false = eval mode, as in `boltz predict`.",
    )
    optimization.add_argument("--distogram_only", type=str2bool, default=False)
    optimization.add_argument("--design_algorithm", choices=["3stages", "3stages_extra"], default="3stages")
    optimization.add_argument("--learning_rate", type=float, default=0.1)
    optimization.add_argument("--learning_rate_pre", type=float, default=1.0)
    optimization.add_argument("--e_soft", type=float, default=0.8)
    optimization.add_argument("--e_soft_1", type=float, default=0.8)
    optimization.add_argument("--e_soft_2", type=float, default=1.0)
    optimization.add_argument("--inter_chain_cutoff", type=float, default=20.0)
    optimization.add_argument("--intra_chain_cutoff", type=float, default=14.0)
    optimization.add_argument("--num_inter_contacts", type=int, default=2)
    optimization.add_argument("--num_intra_contacts", type=int, default=2)
    optimization.add_argument("--optimize_contact_per_binder_pos", type=str2bool, default=True)
    optimization.add_argument("--mask_ligand", type=str2bool, default=False)
    optimization.add_argument("--helix_loss_max", type=float, default=0.0)
    optimization.add_argument("--helix_loss_min", type=float, default=-0.3)

    losses = parser.add_argument_group("loss weights")
    losses.add_argument("--con_loss", type=float, default=1.0)
    losses.add_argument("--i_con_loss", type=float, default=1.0)
    losses.add_argument("--plddt_loss", type=float, default=0.1)
    losses.add_argument("--pae_loss", type=float, default=0.4)
    losses.add_argument("--i_pae_loss", type=float, default=0.1)
    losses.add_argument("--rg_loss", type=float, default=0.0)

    ss = parser.add_argument_group("disulfide geometry")
    ss.add_argument(
        "--use_disulfide_loss", type=str2bool, default=True,
        help="Master switch of the disulfide loss. true (default): enables the distogram Cys-Cys term "
             "(--disulfide_distogram_loss), which carries the pairing to the sequence logits, and the structure terms "
             "(--disulfide_loss and the options of that family), which have no sequence gradient unless --disulfide_grad_steps "
             "or --disulfide_sds_scale is set. false: no disulfide loss at all, in any stage; " + _loss_off_text() + " are "
             "set to 0 and a non-zero value given for one of them is refused. The prescribed Cys pairs still fix the Cys "
             "positions and give the bond feature; --loop_fix_cys, --pre_drop_bonds and --loop_profile are input settings, "
             "not losses, and keep their values. The optional --rc3_defaults true preset keeps the coordinate-based structure terms.",
    )
    ss.add_argument("--disulfide_distance_mode", choices=["hinge", "mse", "bounded"], default="bounded")
    ss.add_argument("--disulfide_target_dist", type=float, default=2.05)
    ss.add_argument("--disulfide_lower_bound", type=float, default=0.0)
    ss.add_argument("--disulfide_upper_bound", type=float, default=2.80)
    ss.add_argument("--disulfide_center_weight", type=float, default=0.0)
    ss.add_argument("--disulfide_hinge_cutoff", type=float, default=2.3)
    ss.add_argument("--disulfide_schedule", type=str2bool, default=True)
    ss.add_argument("--disulfide_hinge_cutoff_start", type=float, default=3.5)
    ss.add_argument("--disulfide_hinge_cutoff_end", type=float, default=2.3)
    ss.add_argument("--disulfide_loss", type=float, default=0.008)
    ss.add_argument("--disulfide_loss_soft", type=float, default=0.004)
    ss.add_argument("--disulfide_loss_temp", type=float, default=0.008)
    ss.add_argument("--disulfide_loss_hard", type=float, default=0.015)
    ss.add_argument("--disulfide_worst_pair_loss", type=float, default=0.5)
    ss.add_argument("--disulfide_filter_cutoff", type=float, default=2.8)
    ss.add_argument("--disulfide_angle_loss", type=float, default=0.0005)
    ss.add_argument("--disulfide_dihedral_loss", type=float, default=0.0002)
    ss.add_argument("--disulfide_angle_target", type=float, default=103.0)
    ss.add_argument("--disulfide_angle_tolerance", type=float, default=25.0)
    ss.add_argument("--disulfide_dihedral_target_abs", type=float, default=90.0)
    ss.add_argument("--disulfide_dihedral_tolerance", type=float, default=45.0)
    ss.add_argument(
        "--disulfide_grad_steps",
        type=int,
        default=0,
        help=("Number of final diffusion steps through which the disulfide geometry loss is "
              "backpropagated to the sequence logits. 0 (default): the geometry terms are computed and logged without a "
              "sequence gradient; the distogram term is not affected. Use --use_disulfide_loss false to disable all "
              "explicit disulfide-loss terms."),
    )
    ss.add_argument(
        "--debug_grad_norm",
        type=str2bool,
        default=False,
        help=("Print, every epoch, the norm of the gradient of the disulfide terms and of all other "
              "terms with respect to the sequence logits (NONE = term has no autograd graph). "
              "Diagnostic only; adds extra backward passes."),
    )
    ss.add_argument("--disulfide_distogram_loss", type=float, default=4.0,
                    help=("Weight of the distogram-based Cys-Cys term: -log P(CB-CB distance of a prescribed pair in the "
                          "bonding window), averaged over the pairs (--disulfide_distogram_form nll). It carries the "
                          "pairing to the sequence logits at every stage, without structure sampling. 0 disables this term."))
    ss.add_argument("--disulfide_sds_scale", type=float, default=0.0,
                    help=("Weight multiplier for disulfide geometry terms evaluated on auxiliary denoiser estimates. "
                          "0 (default) disables these terms. Needs --disulfide_loss > 0."))
    ss.add_argument("--disulfide_sds_samples", type=int, default=2,
                    help="Noise levels (denoiser passes with autograd) per epoch for the SDS terms; memory grows with it.")
    ss.add_argument("--disulfide_sds_sigma_lo", type=float, default=1.5, help="Lowest SDS noise level, Angstrom.")
    ss.add_argument("--disulfide_sds_sigma_hi", type=float, default=6.0, help="Highest SDS noise level, Angstrom.")
    ss.add_argument("--disulfide_sds_clip", type=float, default=2.0,
                    help=("SG-SG violation (Angstrom beyond the upper bound) above which a pair stops contributing "
                          "to the SDS distance terms (structure too far off to be informative)."))
    ss.add_argument("--debug_sds_profile_every", type=int, default=0,
                    help=("Diagnostic: every N-th epoch (pre-run excluded) print the '[sds_profile]' noise-level "
                          "profile of the denoiser-estimate disulfide gradient (extra backward passes; 0 = off)."))
    ss.add_argument("--debug_sds_profile_t", type=str, default="1.5,3,6,12,24,48",
                    help="Noise levels (Angstrom, comma separated) of the --debug_sds_profile_every profile.")
    ss.add_argument("--disulfide_distogram_lo", type=float, default=3.3)
    ss.add_argument("--disulfide_distogram_hi", type=float, default=4.8)
    ss.add_argument("--disulfide_distogram_form", choices=["prob", "nll"], default="nll",
                    help=("Form of the distogram Cys-Cys term: nll = -log P(window) (non-saturating); prob = 1 - P(window) "
                          "(its gradient vanishes when P(window) is small)."))
    ss.add_argument(
        "--rc3_defaults", type=str2bool, default=False,
        help="Apply the optional input/loss preset: " + _rc3_values_text() + ". "
             "Options given explicitly on the command line still win.",
    )
    # --rc3_defaults and --use_disulfide_loss are read first and only change defaults, so it does not matter where they stand on
    # the command line
    first = parser.parse_known_args()[0]
    if first.rc3_defaults:
        parser.set_defaults(**RC3_DEFAULTS)
    if not first.use_disulfide_loss:
        parser.set_defaults(**LOSS_OFF)
    args = parser.parse_args()
    if not args.use_disulfide_loss:
        asked = [f"--{key} {getattr(args, key):g}" for key, off in LOSS_OFF.items() if getattr(args, key) != off]
        if asked:
            parser.error("--use_disulfide_loss false means no disulfide loss at all, but " + ", ".join(asked)
                         + " asks for one. Remove it, or keep --use_disulfide_loss true.")
    return args


def _parse_floats(text):
    try:
        vals = tuple(float(p) for p in str(text).replace(' ', '').split(',') if p != '')
    except ValueError:
        vals = ()
    if not vals or min(vals) <= 0:
        raise SystemExit("--debug_sds_profile_t needs positive numbers separated by commas, e.g. 1.5,3,6,12,24,48")
    return vals


# The design engine finds the designed chain by its entity index, through a table that starts with 'A' = 0. The generated input YAML
# always lists the binder first, so the binder is entity 0 whatever output chain ID --binder_chain gives it; the ID only names
# the chain in the YAML and in the predictions. The engine is therefore always asked for the first entity.
ENGINE_BINDER_KEY = "A"


def load_config(args: argparse.Namespace) -> dict:
    config_path = Path(__file__).with_name("configs") / "default_protein_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update({
        "binder_chain": ENGINE_BINDER_KEY,
        "length_min": args.binder_length,
        "length_max": args.binder_length,
        "msa_max_seqs": args.msa_max_seqs,
        "pocket_conditioning": bool(args.contact_residues),
        "mask_ligand": args.mask_ligand,
        "optimize_contact_per_binder_pos": args.optimize_contact_per_binder_pos,
        "distogram_only": args.distogram_only,
        "design_algorithm": args.design_algorithm,
        "learning_rate": args.learning_rate,
        "learning_rate_pre": args.learning_rate_pre,
        "e_soft": args.e_soft,
        "e_soft_1": args.e_soft_1,
        "e_soft_2": args.e_soft_2,
        "inter_chain_cutoff": args.inter_chain_cutoff,
        "intra_chain_cutoff": args.intra_chain_cutoff,
        "num_inter_contacts": args.num_inter_contacts,
        "num_intra_contacts": args.num_intra_contacts,
        "helix_loss_max": args.helix_loss_max,
        "helix_loss_min": args.helix_loss_min,
        "optimizer_type": args.optimizer_type,
        "pre_iteration": args.pre_iteration,
        "soft_iteration": args.soft_iteration,
        "temp_iteration": args.temp_iteration,
        "hard_iteration": args.hard_iteration,
        "semi_greedy_steps": args.semi_greedy_steps,
        "recycling_steps": args.recycling_steps,
        "set_train": args.set_train,
        "loop_fix_cys": args.loop_fix_cys,
        "loop_drop_bonds": args.loop_drop_bonds,
        "pre_drop_bonds": args.pre_drop_bonds,
        "loop_profile": args.loop_profile,
        "seed": args.seed,
    })
    return config


def resolve_run_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    """Resolve user paths against the launch directory before changing CWD.

    Relative ``--target_pdb``, ``--checkpoint``, ``--ccd_path`` and
    ``--work_dir`` values are interpreted relative to the directory from
    which ``highss`` was invoked.
    """
    launch_dir = Path.cwd().resolve()

    def resolve_from_launch(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = launch_dir / path
        return path.resolve()

    target_pdb = resolve_from_launch(args.target_pdb)
    checkpoint = resolve_from_launch(args.checkpoint)
    ccd_path = resolve_from_launch(args.ccd_path)
    work_dir = resolve_from_launch(args.work_dir)

    for label, path in (
        ("target PDB", target_pdb),
        ("checkpoint", checkpoint),
        ("CCD", ccd_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    return target_pdb, checkpoint, ccd_path, work_dir


def main() -> None:
    args = parse_arguments()

    # Resolve every user-supplied path before changing into --work_dir.
    target_pdb, checkpoint_path, ccd_path_obj, work_dir = resolve_run_paths(args)

    # Delay heavy Boltz/model imports until an actual design is started.
    from .design_engine import get_boltz_model, run_highss_design

    if args.binder_length < 4:
        raise ValueError("Binder length must be at least 4 residues.")

    target_name = args.target_name.strip() or target_pdb.stem
    input_chains = [x.strip() for x in args.target_chains.split(",") if x.strip()]
    if not input_chains:
        raise ValueError("At least one target chain is required.")
    chain_sequences = get_chains_sequence(target_pdb)
    missing = [chain for chain in input_chains if chain not in chain_sequences]
    if missing:
        raise ValueError(f"Target chain(s) not found in PDB: {missing}")
    target_sequences = [chain_sequences[chain] for chain in input_chains]

    contact_target_index = None
    contact_chain = ""
    contact_positions = ""
    if args.contact_residues:
        selected = args.contact_target_chain.strip()
        if not selected:
            if len(input_chains) != 1:
                raise ValueError("Use --contact_target_chain for a multi-chain target.")
            selected = input_chains[0]
        if selected not in input_chains:
            raise ValueError("--contact_target_chain must be present in --target_chains.")
        contact_target_index = input_chains.index(selected)
        # --contact_residues are residue numbers of the PDB file; the pocket constraint counts positions along the chain sequence.
        contact_chain = selected
        contact_positions = ",".join(str(position) for position in resolve_contact_residues(
            args.contact_residues, get_chain_residue_labels(target_pdb)[selected], selected))

    pairs = parse_disulfide_pairs(args.disulfide_pairs, args.binder_length)
    binder_bonds, fix_residues = disulfide_cli_fields(pairs)

    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    input_config = InputConfig(
        work_dir / "inputs" / f"protein_{target_name}_{args.suffix}"
    )
    input_config.setup_directories()
    _, yaml_path, target_output_chains = generate_protein_binder_yaml(
        target_name=target_name,
        target_sequences=target_sequences,
        binder_chain=args.binder_chain,
        binder_length=args.binder_length,
        disulfide_pairs=pairs,
        config=input_config,
        msa_mode=args.msa_mode,
        msa_max_seqs=args.msa_max_seqs,
        contact_target_index=contact_target_index,
        contact_residues=contact_positions,
    )

    checkpoint = str(checkpoint_path)
    ccd_path = str(ccd_path_obj)
    boltz_path = shutil.which("boltz")
    if boltz_path is None:
        raise FileNotFoundError("The 'boltz' command is not available. Run ./setup.sh first.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    if args.disulfide_sds_scale > 0:
        if args.disulfide_loss <= 0:
            raise SystemExit("--disulfide_sds_scale needs --disulfide_loss > 0 (the SDS terms reuse its weights)")
        if args.disulfide_sds_samples < 1 or not (0.0 < args.disulfide_sds_sigma_lo <= args.disulfide_sds_sigma_hi):
            raise SystemExit("--disulfide_sds_samples >= 1 and 0 < --disulfide_sds_sigma_lo <= --disulfide_sds_sigma_hi needed")
    if args.debug_sds_profile_every > 0:
        _parse_floats(args.debug_sds_profile_t)
    model = get_boltz_model(checkpoint, predict_args, device)
    if not getattr(model, "confidence_prediction", True):
        raise ValueError(
            "The checkpoint has no confidence head. Design is driven by pLDDT, PAE and iPTM, so it needs a checkpoint that "
            "includes the head, such as the released com-FT.ckpt (see checkpoints/README.md)."
        )
    model.train()
    config = load_config(args)
    version_name = f"highss_{target_name}_{args.suffix}"
    output_root = work_dir / "outputs"
    output_root.mkdir(exist_ok=True)

    print("HighSS run")
    print(f"  target PDB:       {target_pdb}")
    print(f"  input chains:     {','.join(input_chains)}")
    print(f"  output chains:    {','.join(target_output_chains)}")
    print(f"  binder chain:     {args.binder_chain}")
    print(f"  binder length:    {args.binder_length}")
    print(f"  disulfide pairs:  {args.disulfide_pairs}")
    print(f"  binder bonds:     {binder_bonds}")
    if contact_positions:
        print(f"  contact residues: chain {contact_chain} PDB {args.contact_residues} -> sequence positions {contact_positions}")
    print(f"  MSA mode:         {args.msa_mode}")
    print(f"  YAML:             {yaml_path}")
    print(f"  output version:   {version_name}")
    term = (f"disulfide loss ON; distogram term {args.disulfide_distogram_form} x {args.disulfide_distogram_loss:g}"
            + ("; geometry backprop ON" if args.disulfide_grad_steps > 0 else "") if args.use_disulfide_loss
            else "disulfide loss OFF (no disulfide term in any stage)")
    print(f"  loop guidance:    {term}; "
          f"loop_fix_cys {args.loop_fix_cys}; pre_drop_bonds {args.pre_drop_bonds}"
          f"; loop_profile {args.loop_profile}"
          + ("  (--rc3_defaults: optional preset; explicit options override)" if args.rc3_defaults else ""))

    loss_scales = {
        "con_loss": args.con_loss,
        "i_con_loss": args.i_con_loss,
        "plddt_loss": args.plddt_loss,
        "pae_loss": args.pae_loss,
        "i_pae_loss": args.i_pae_loss,
        "rg_loss": args.rg_loss,
        "disulfide_loss": args.disulfide_loss,
        "disulfide_worst_pair_loss": args.disulfide_worst_pair_loss,
        "disulfide_distogram_loss": args.disulfide_distogram_loss,
        "disulfide_angle_loss": args.disulfide_angle_loss,
        "disulfide_dihedral_loss": args.disulfide_dihedral_loss,
    }
    run_highss_design(
        boltz_path=boltz_path,
        main_dir=str(output_root),
        yaml_dir=str(yaml_path.parent),
        boltz_model=model,
        ccd_path=ccd_path,
        design_samples=args.design_samples,
        version_name=version_name,
        config=config,
        loss_scales=loss_scales,
        show_animation=args.show_animation,
        save_trajectory=args.save_trajectory,
        redo_boltz_predict=args.redo_boltz_predict,
        checkpoint=checkpoint,
        fix_residues=fix_residues,
        binder_bonds=binder_bonds,
        residue_index_base=1,
        disulfide_target_dist=args.disulfide_target_dist,
        disulfide_hinge_cutoff=args.disulfide_hinge_cutoff,
        disulfide_distance_mode=args.disulfide_distance_mode,
        disulfide_lower_bound=args.disulfide_lower_bound,
        disulfide_upper_bound=args.disulfide_upper_bound,
        disulfide_center_weight=args.disulfide_center_weight,
        disulfide_schedule=args.disulfide_schedule,
        disulfide_hinge_cutoff_start=args.disulfide_hinge_cutoff_start,
        disulfide_hinge_cutoff_end=args.disulfide_hinge_cutoff_end,
        disulfide_loss_soft=args.disulfide_loss_soft,
        disulfide_loss_temp=args.disulfide_loss_temp,
        disulfide_loss_hard=args.disulfide_loss_hard,
        disulfide_filter_cutoff=args.disulfide_filter_cutoff,
        disulfide_angle_target=args.disulfide_angle_target,
        disulfide_angle_tolerance=args.disulfide_angle_tolerance,
        disulfide_dihedral_target_abs=args.disulfide_dihedral_target_abs,
        disulfide_dihedral_tolerance=args.disulfide_dihedral_tolerance,
        disulfide_grad_steps=args.disulfide_grad_steps,
        debug_grad_norm=args.debug_grad_norm,
        disulfide_sds_scale=args.disulfide_sds_scale,
        disulfide_sds_samples=args.disulfide_sds_samples,
        disulfide_sds_sigma_lo=args.disulfide_sds_sigma_lo,
        disulfide_sds_sigma_hi=args.disulfide_sds_sigma_hi,
        disulfide_sds_clip=args.disulfide_sds_clip,
        disulfide_sds_profile_every=args.debug_sds_profile_every,
        disulfide_sds_profile_t=_parse_floats(args.debug_sds_profile_t),
        disulfide_distogram_lo=args.disulfide_distogram_lo,
        disulfide_distogram_hi=args.disulfide_distogram_hi,
        disulfide_distogram_form=args.disulfide_distogram_form,
    )
    print("HighSS design completed successfully.")


if __name__ == "__main__":
    main()
