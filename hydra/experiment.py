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
    parser = parsers.COCOMaskParser(annotations_filepath=cfg.annotations_filepath, img_dir=cfg.img_dir)
    train_records, valid_records = parser.parse()
    class_map = cfg.datamodule.class_map

    # MODEL
    model_type = cfg.model.model_name
    backbone = cfg.backbones.backbone_name

    """
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

    """

    # TRAINER
    train_tfms = tfms.A.Adapter(
    [
        tfms.A.Normalize(),
    ]
    )

    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=cfg.datamodule.chip_sz), tfms.A.Normalize()])

    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    train_dl = model_type.train_dl(train_ds, batch_size=cfg.datamodule.bs, num_workers=cfg.trainer.num_workers, shuffle=True) # adjust num_workers for your processor count
    valid_dl = model_type.valid_dl(valid_ds, batch_size=cfg.datamodule.bs, num_workers=cfg.trainer.num_workers, shuffle=False)

    infer_dl = model_type.infer_dl(valid_ds, batch_size=cfg.datamodule.bs, shuffle=False)

    model = model_type.model(backbone=backbone(pretrained=cfg.backbones.pretrained), num_classes=len(parser.class_map))

    metrics = [cfg.trainer.metric(metric_type=cfg.trainer.metric_type)]

    learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)

    # FIT
    learn.fine_tune(cfg.trainer.num_epochs, cfg.trainer.lr)



if __name__ == "__main__":
    train()