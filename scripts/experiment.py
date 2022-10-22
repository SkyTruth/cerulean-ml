import os
from datetime import datetime
from pathlib import Path

import hydra
from fastai.callback.tracker import SaveModelCallback
from icevision import Dataset, models, tfms
from icevision.data import *
from icevision.data import Dataset
from icevision.imports import *
from icevision.metrics import (  # make sure you have the SkyTruth fork of icevision installed
    SimpleConfusionMatrix,
)
from icevision.metrics.metric import *
from icevision.utils import *
from omegaconf import DictConfig

from ceruleanml import preprocess


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    cfg.name = (
        cfg.name
        if cfg.name is not None
        else f"{cfg.author}-{cfg.model.model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    print(cfg.name)
    # Create logs/ - wandb logs to an already existing directory only
    cwd = os.getcwd()
    (Path(cwd) / "logs").mkdir(exist_ok=True)
    negative_sample_count = 0
    area_thresh = 0
    classes_to_remove=[
        "ambiguous",
        ]
    classes_to_remap ={
        # "old_vessel": "recent_vessel",
        # "coincident_vessel": "recent_vessel",
    }

    train_records = preprocess.load_set_record_collection(
        cfg.datamodule.annotations_filepath,
        cfg.datamodule.img_dir,
        area_thresh,
        negative_sample_count,
        preprocess=False,
        classes_to_remap=classes_to_remap, 
        classes_to_remove=classes_to_remove,
        classes_to_keep=classes_to_keep,
    )

    # MODEL
    icevision_model = models.torchvision.mask_rcnn
    backbone = icevision_model.backbones.resnet34_fpn
    # TRAINER
    train_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=cfg.trainer.resize)])

    # valid_tfms = tfms.A.Adapter(
    #     [*tfms.A.resize_and_pad(size=cfg.datamodule.chip_sz), tfms.A.Normalize()]
    # )

    train_ds = Dataset(train_records, train_tfms)

    train_dl = icevision_model.train_dl(
        train_ds,
        batch_size=cfg.datamodule.bs,
        num_workers=cfg.trainer.num_workers,
        shuffle=True,
    )  # adjust num_workers for your processor count

    model = icevision_model.model(
        backbone=backbone(pretrained=cfg.model.pretrained),
        num_classes=len(data.class_list),
    )
    metrics = [SimpleConfusionMatrix(print_summary=True)]
    learn = icevision_model.fastai.learner(
        dls=[train_dl, train_dl],
        model=model,
        cbs=SaveModelCallback(min_delta=0.01),
        metrics=metrics,
    )

    # FIT
    learn.fine_tune(cfg.trainer.num_epochs, cfg.trainer.lr)


if __name__ == "__main__":
    train()
