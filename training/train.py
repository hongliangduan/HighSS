"""Boltz-1 fine-tuning entry point used for the HighSS release package."""

from __future__ import annotations

import os
import random
import shutil
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, listconfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from boltz.data.module.training import BoltzTrainingDataModule, DataConfig


@dataclass
class TrainConfig:
    data: DataConfig
    model: LightningModule
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    wandb: Optional[dict] = None
    disable_checkpoint: bool = False
    matmul_precision: Optional[str] = None
    find_unused_parameters: Optional[bool] = False
    save_top_k: Optional[int] = 1
    validation_only: bool = False
    debug: bool = False
    strict_loading: bool = True
    load_confidence_from_trunk: Optional[bool] = False
    seed: int = 42
    checkpoint_monitor: str = "val/best_rmsd"
    checkpoint_mode: str = "min"
    release_checkpoint_name: Optional[str] = None


def _require_path(path_value: Optional[str], label: str) -> None:
    if path_value in (None, "", "null"):
        return
    if not Path(path_value).exists():
        raise FileNotFoundError(f"{label} does not exist: {path_value}")


def _validate_inputs(raw_config: omegaconf.DictConfig) -> None:
    pretrained = raw_config.get("pretrained")
    resume = raw_config.get("resume")
    if resume:
        _require_path(str(resume), "resume checkpoint")
    elif pretrained:
        _require_path(str(pretrained), "pretrained checkpoint")

    for index, dataset in enumerate(raw_config.data.datasets):
        _require_path(str(dataset.target_dir), f"dataset {index} target_dir")
        if dataset.get("msa_dir"):
            _require_path(str(dataset.msa_dir), f"dataset {index} msa_dir")
        if dataset.get("split"):
            _require_path(str(dataset.split), f"dataset {index} validation split")


def train(raw_config_path: str, args: list[str]) -> None:
    raw_config = omegaconf.OmegaConf.load(raw_config_path)
    overrides = omegaconf.OmegaConf.from_dotlist(args)
    raw_config = omegaconf.OmegaConf.merge(raw_config, overrides)

    _validate_inputs(raw_config)
    cfg = TrainConfig(**hydra.utils.instantiate(raw_config))

    pl.seed_everything(cfg.seed, workers=True)
    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    output_dir = Path(cfg.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_args = dict(cfg.trainer or {})
    devices = trainer_args.get("devices", 1)
    wandb_cfg = cfg.wandb

    if cfg.debug:
        devices = 1 if isinstance(devices, int) else [devices[0]]
        trainer_args["devices"] = devices
        cfg.data.num_workers = 0
        wandb_cfg = None

    data_module = BoltzTrainingDataModule(DataConfig(**cfg.data))
    model_module = cfg.model

    if cfg.pretrained and not cfg.resume:
        if cfg.load_confidence_from_trunk:
            checkpoint = torch.load(cfg.pretrained, map_location="cpu")
            new_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("structure_module") and not key.startswith(
                    "distogram_module"
                ):
                    new_state_dict[f"confidence_module.{key}"] = value
            new_state_dict.update(checkpoint["state_dict"])
            checkpoint["state_dict"] = new_state_dict
            random_string = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=10)
            )
            temporary_checkpoint = Path(cfg.pretrained).with_name(
                f"{random_string}.ckpt"
            )
            torch.save(checkpoint, temporary_checkpoint)
            checkpoint_path = str(temporary_checkpoint)
        else:
            temporary_checkpoint = None
            checkpoint_path = cfg.pretrained

        print(f"Loading pretrained weights from {checkpoint_path}")
        model_module = type(model_module).load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            strict=False,
            **model_module.hparams,
        )
        if temporary_checkpoint is not None:
            temporary_checkpoint.unlink(missing_ok=True)

    callbacks: list[ModelCheckpoint] = []
    best_callback: Optional[ModelCheckpoint] = None
    if not cfg.disable_checkpoint:
        epoch_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename="model-epoch{epoch:02d}",
            save_top_k=cfg.save_top_k,
            every_n_epochs=1,
            auto_insert_metric_name=False,
        )
        best_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename="best-model-epoch{epoch:02d}",
            monitor=cfg.checkpoint_monitor,
            save_top_k=1,
            save_last=True,
            mode=cfg.checkpoint_mode,
            every_n_epochs=1,
            auto_insert_metric_name=False,
        )
        callbacks = [epoch_callback, best_callback]

    loggers = []
    if wandb_cfg:
        wandb_logger = WandbLogger(
            name=wandb_cfg["name"],
            save_dir=cfg.output,
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            log_model=False,
        )
        loggers.append(wandb_logger)

        @rank_zero_only
        def save_config_to_wandb() -> None:
            config_out = Path(wandb_logger.experiment.dir) / "run.yaml"
            OmegaConf.save(raw_config, config_out)
            wandb_logger.experiment.save(str(config_out))

        save_config_to_wandb()

    strategy: object = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, (list, listconfig.ListConfig)) and len(devices) > 1
    ):
        strategy = DDPStrategy(find_unused_parameters=cfg.find_unused_parameters)

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=not cfg.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        **trainer_args,
    )

    if not cfg.strict_loading:
        model_module.strict_loading = False

    if cfg.validation_only:
        trainer.validate(model_module, datamodule=data_module, ckpt_path=cfg.resume)
        return

    trainer.fit(model_module, datamodule=data_module, ckpt_path=cfg.resume)

    if (
        trainer.is_global_zero
        and cfg.release_checkpoint_name
        and best_callback is not None
        and best_callback.best_model_path
    ):
        release_path = output_dir / cfg.release_checkpoint_name
        shutil.copy2(best_callback.best_model_path, release_path)
        print(f"Release checkpoint copied to {release_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts/train/train.py <config.yaml> [key=value ...]"
        )
    train(sys.argv[1], sys.argv[2:])
