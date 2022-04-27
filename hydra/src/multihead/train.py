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

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    model_type = cfg.model.model_name
    backbone = cfg.backbones.backbone_name
    parser = parsers.COCOMaskParser(annotations_filepath=cfg.annotations_filepath, img_dir=cfg.img_dir)
    train_records, valid_records = parser.parse()
    class_map = cfg.datamodule.class_map

    train_tfms = tfms.A.Adapter(
    [
        tfms.A.Normalize(),
    ]
    )

    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=512), tfms.A.Normalize()])

    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    train_dl = model_type.train_dl(train_ds, batch_size=cfg.datamodule.bs, num_workers=6, shuffle=True) # adjust num_workers for your processor count
    valid_dl = model_type.valid_dl(valid_ds, batch_size=cfg.datamodule.bs, num_workers=6, shuffle=False)

    infer_dl = model_type.infer_dl(valid_ds, batch_size=cfg.datamodule.bs, shuffle=False)

    model = model_type.model(backbone=backbone(pretrained=cfg.backbones.pretrained), num_classes=len(parser.class_map))

    metrics = [IoUMetric(metric_type=IoUMetricType.mask)]

    learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)
    learn.fine_tune(cfg.hparams.num_epochs, cfg.hparams.lr)

    return


if __name__ == "__main__":
    train()
