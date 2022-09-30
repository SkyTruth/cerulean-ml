import os
import random
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.transforms import IndexSplitter, Normalize
from fastai.losses import DiceLoss
from fastai.metrics import Dice, DiceMulti
from fastai.vision.augment import Resize, aug_transforms
from fastai.vision.core import AddMaskCodes, PILImageBW
from fastai.vision.data import ImageBlock, MaskBlock
from fastai.vision.learner import unet_learner
from icevision.visualize import show_data
from torchvision.models import resnet18, resnet34, resnet50

from ceruleanml import data, evaluation, preprocess
from ceruleanml.coco_load_fastai import (
    get_image_path,
    record_collection_to_record_ids,
    record_to_mask,
)
from ceruleanml.multispectral_blocks import (
    MSTensorImage,
    MSTensorMask,
    TensorImageResizer,
    open_mask_from_record,
    open_n_channel_img,
)

with_context = False
mount_path = "/root/"
train_set = "train-no-context-512"
tiled_images_folder_train = "tiled_images_no_context"
json_name_train = "instances_TiledCeruleanDatasetV2NoContextFiles.json"

coco_json_path_train = f"{mount_path}/partitions/{train_set}/{json_name_train}"
tiled_images_folder_train = f"{mount_path}/partitions/{train_set}/{tiled_images_folder_train}"
val_set = "val-no-context-512"
tiled_images_folder_val = "tiled_images_no_context"
json_name_val = "instances_TiledCeruleanDatasetV2NoContextFiles.json"
coco_json_path_val = f"{mount_path}/partitions/{val_set}/{json_name_val}"
tiled_images_folder_val = f"{mount_path}/partitions/{val_set}/{tiled_images_folder_val}"

# with aux files
# with_context=True
# mount_path = "/root/"
# train_set = "train-with-context-512"
# tiled_images_folder_train = "tiled_images"
# json_name_train = "instances_TiledCeruleanDatasetV2.json"

# coco_json_path_train = f"{mount_path}/partitions/{train_set}/{json_name_train}"
# tiled_images_folder_train = f"{mount_path}/partitions/{train_set}/{tiled_images_folder_train}"
# val_set = "val-with-context-512"
# tiled_images_folder_val= "tiled_images"
# json_name_val = "instances_TiledCeruleanDatasetV2.json"
# coco_json_path_val= f"{mount_path}/partitions/{val_set}/{json_name_val}"
# tiled_images_folder_val = f"{mount_path}/partitions/{val_set}/{tiled_images_folder_val}"

negative_sample_count = 0
negative_sample_count_val = 40
area_thresh = 10
# f"{mount_path}/partitions/val/instances_tiled_cerulean_train_v2.json"

record_collection_train = preprocess.load_set_record_collection(
    coco_json_path_train,
    tiled_images_folder_train,
    area_thresh,
    negative_sample_count,
    preprocess=True,
)
record_ids_train = record_collection_to_record_ids(record_collection_train)

record_collection_val = preprocess.load_set_record_collection(
    coco_json_path_val,
    tiled_images_folder_val,
    area_thresh,
    negative_sample_count_val,
    preprocess=True,
)
record_ids_val = record_collection_to_record_ids(record_collection_val)


train_val_record_ids = record_ids_train + record_ids_val
combined_record_collection = record_collection_train + record_collection_val


def get_val_indices(combined_ids, val_ids):
    return list(range(len(combined_ids)))[-len(val_ids) :]


val_indices = get_val_indices(train_val_record_ids, record_ids_val)


size = 128  # Progressive resizing could happen here


def get_image_by_record_id(record_id):
    return get_image_path(combined_record_collection, record_id)


def get_mask_by_record_id(record_id):
    return record_to_mask(combined_record_collection, record_id)


def create_data_block(size, with_context=True):
    imblock = ImageBlock if with_context else ImageBlock(cls=PILImageBW)

    coco_seg_dblock = DataBlock(
        blocks=(
            imblock,
            MaskBlock(codes=data.class_list),
        ),  # ImageBlock is RGB by default, uses PIL
        get_x=get_image_by_record_id,
        splitter=IndexSplitter(val_indices),
        get_y=get_mask_by_record_id,  # *aug_transforms(),  # we need to normalize here or else fastai incorrectly normalizes by the pretrained stats
        item_tfms=Resize(size),
        n_inp=1,
    )
    return coco_seg_dblock


coco_seg_dblock = create_data_block(size, with_context=with_context)
dls = coco_seg_dblock.dataloaders(source=train_val_record_ids, batch_size=6)


db_custom = DataBlock(
    blocks=(
        TransformBlock(
            type_tfms=partial(
                MSTensorImage.create,
                chnls=[0],
                record_collection=combined_record_collection,
            )
        ),
        TransformBlock(
            type_tfms=partial(MSTensorMask.create, record_collection=combined_record_collection),
            item_tfms=AddMaskCodes(codes=data.class_list),
        ),
    ),
    get_items=record_collection_to_record_ids,
    splitter=IndexSplitter(val_indices),
    n_inp=1,
    item_tfms=TensorImageResizer(size),
    batch_tfms=[Normalize.from_stats([60.73], [16.099])],
)

dls_custom = db_custom.dataloaders(source=combined_record_collection, bs=6)


dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S")
experiment_dir = Path(f"{mount_path}/experiments/cv2/" + timestampStr + "_fastai_unet/")
experiment_dir.mkdir(exist_ok=True)
print(experiment_dir)

arch = 18
archs = {18: resnet18, 34: resnet34, 50: resnet50}

# removed these callbacks since they cause this error: https://forums.fast.ai/t/learner-object-has-no-attribute-recorder/46328/18
# or no model.pth found in experiment folder for some reason
cbs = [  # SaveModelCallback(monitor="valid_loss", with_opt=True),
    # EarlyStoppingCallback(monitor='valid_loss', min_delta=0.005, patience=5),
    # TensorBoardCallback(projector=False, trace_model=False)
]

learner = unet_learner(
    dls,
    archs[arch],
    metrics=[DiceMulti()],
    model_dir=experiment_dir,
    n_in=1,
    cbs=cbs,
    normalize=False,
    pretrained=False,
)
