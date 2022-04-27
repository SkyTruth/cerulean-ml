import os
from datetime import datetime
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
from icevision import models, parsers, show_records, tfms, Dataset, Metric, COCOMetric, COCOMetricType
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
 
from icevision.imports import *
from icevision.utils import *
from icevision.data import *
from icevision.metrics.metric import *

import numpy as np

import json
import os
from PIL import Image
from pycococreatortools import pycococreatortools

from src.multihead.metrics import IoUMetric, IoUMetricType
from src.multihead.parsers import CMParser
from src.multihead.model import Model


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    cfg.name = (
        cfg.name
        if cfg.name is not None
        else f"{cfg.author}-{cfg.model.model_name}-{cfg.optimizer.name}-{cfg.loss.name}-mixup-{cfg.mixup}-{cfg.datamodule.label}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    print(cfg.name)
    # Create logs/ - wandb logs to an already existing directory only
    cwd = os.getcwd()
    (Path(cwd) / "logs").mkdir(exist_ok=True)

    # DATAMODULE
    dm = MapillaryDataModule(cfg.datamodule)

    # MODEL
    model = Model(cfg)

    # LOGGERS
    wandb_logger = WandbLogger(
        project="cerulean",
        save_dir="logs",
        name=cfg.name,
        log_model=False,
    )
    # Wandb Isuue with PL: Replaces `name` with `project` that causes over-writing of params in the dashboard

    csv_logger = CSVLogger(save_dir="logs", name=cfg.name)

    # CALLBACKS
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # TRAINER
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        #callbacks=callbacks,
        #logger=[wandb_logger, csv_logger],
        #_convert_="partial",
    )

    # FIT
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    train()