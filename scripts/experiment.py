import os
from datetime import datetime
from pathlib import Path

import hydra
from icevision import models, parsers, tfms
from icevision.data import Dataset, SingleSplitSplitter
from omegaconf import DictConfig

from ceruleanml.metrics import IoUMetric, IoUMetricType


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

    # DATAMODULE
    parser = parsers.COCOMaskParser(
        annotations_filepath=cfg.datamodule.annotations_filepath,
        img_dir=cfg.datamodule.img_dir,
    )

    train_records = parser.parse(data_splitter=SingleSplitSplitter(), autofix=False)

    # MODEL
    icevision_model = models.torchvision.mask_rcnn
    backbone = icevision_model.backbones.resnet34_fpn
    # TRAINER
    train_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=cfg.trainer.resize)])

    # valid_tfms = tfms.A.Adapter(
    #     [*tfms.A.resize_and_pad(size=cfg.datamodule.chip_sz), tfms.A.Normalize()]
    # )

    train_ds = Dataset(train_records[0], train_tfms)

    train_dl = icevision_model.train_dl(
        train_ds,
        batch_size=cfg.datamodule.bs,
        num_workers=cfg.trainer.num_workers,
        shuffle=True,
    )  # adjust num_workers for your processor count

    model = icevision_model.model(
        backbone=backbone(pretrained=cfg.model.pretrained),
        num_classes=len(parser.class_map),
    )
    metrics = [IoUMetric(metric_type=IoUMetricType.mask)]

    learn = icevision_model.fastai.learner(
        dls=[train_dl, train_dl], model=model, metrics=metrics
    )

    # FIT
    learn.fine_tune(cfg.trainer.num_epochs, cfg.trainer.lr)


if __name__ == "__main__":
    train()
