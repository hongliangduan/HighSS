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
    get_chains_sequence,
    parse_disulfide_pairs,
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
        default="none",
        help="none=single sequence; target=MSA only for target chains, never the binder",
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
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config_path = Path(__file__).with_name("configs") / "default_protein_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update({
        "binder_chain": args.binder_chain,
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
    if args.contact_residues:
        selected = args.contact_target_chain.strip()
        if not selected:
            if len(input_chains) != 1:
                raise ValueError("Use --contact_target_chain for a multi-chain target.")
            selected = input_chains[0]
        if selected not in input_chains:
            raise ValueError("--contact_target_chain must be present in --target_chains.")
        contact_target_index = input_chains.index(selected)

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
        contact_residues=args.contact_residues,
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
    model = get_boltz_model(checkpoint, predict_args, device)
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
    print(f"  MSA mode:         {args.msa_mode}")
    print(f"  YAML:             {yaml_path}")
    print(f"  output version:   {version_name}")

    loss_scales = {
        "con_loss": args.con_loss,
        "i_con_loss": args.i_con_loss,
        "plddt_loss": args.plddt_loss,
        "pae_loss": args.pae_loss,
        "i_pae_loss": args.i_pae_loss,
        "rg_loss": args.rg_loss,
        "disulfide_loss": args.disulfide_loss,
        "disulfide_worst_pair_loss": args.disulfide_worst_pair_loss,
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
    )
    print("HighSS design completed successfully.")


if __name__ == "__main__":
    main()
