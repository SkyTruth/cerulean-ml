from datetime import datetime
from pathlib import Path

from fastai.callback.tensorboard import TensorBoardCallback
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.data.block import DataBlock
from fastai.data.transforms import IndexSplitter
from fastai.metrics import Dice, DiceMulti
from fastai.vision.augment import aug_transforms
from fastai.vision.data import ImageBlock, MaskBlock
from fastai.vision.learner import unet_learner
from torchvision.models import resnet18, resnet34, resnet50

from ceruleanml import data, evaluation, preprocess
from ceruleanml.coco_load_fastai import (
    get_image_path,
    record_collection_to_record_ids,
    record_to_mask,
)
from ceruleanml.inference import save_fastai_model_state_dict_and_tracing

# Parsing COCO Dataset with Icevision

with_context = False
mount_path = "/root/"
train_set = "train-no-context-512"
tiled_images_folder_train = "tiled_images_no_context"
json_name_train = "instances_TiledCeruleanDatasetV2NoContextFiles.json"

coco_json_path_train = f"{mount_path}/partitions/{train_set}/{json_name_train}"
tiled_images_folder_train = (
    f"{mount_path}/partitions/{train_set}/{tiled_images_folder_train}"
)
val_set = "val-no-context-512"
tiled_images_folder_val = "tiled_images_no_context"
json_name_val = "instances_TiledCeruleanDatasetV2NoContextFiles.json"
coco_json_path_val = f"{mount_path}/partitions/{val_set}/{json_name_val}"
tiled_images_folder_val = f"{mount_path}/partitions/{val_set}/{tiled_images_folder_val}"

bs = 8  # max
size = 512
n = "all"
arch = 34
epochs = 100

negative_sample_count = 0
negative_sample_count_val = 0
area_thresh = 10

record_collection_with_negative_small_filtered_train = (
    preprocess.load_set_record_collection(
        coco_json_path_train,
        tiled_images_folder_train,
        area_thresh,
        negative_sample_count,
        preprocess=False,
    )
)
record_ids_train = record_collection_to_record_ids(
    record_collection_with_negative_small_filtered_train
)

record_collection_with_negative_small_filtered_val = (
    preprocess.load_set_record_collection(
        coco_json_path_val,
        tiled_images_folder_val,
        area_thresh,
        negative_sample_count_val,
        preprocess=False,
    )
)
record_ids_val = record_collection_to_record_ids(
    record_collection_with_negative_small_filtered_val
)

assert len(set(record_ids_train)) + len(set(record_ids_val)) == len(
    record_ids_train
) + len(record_ids_val)

train_val_record_ids = record_ids_train + record_ids_val
combined_record_collection = (
    record_collection_with_negative_small_filtered_train
    + record_collection_with_negative_small_filtered_val
)


def get_val_indices(combined_ids, val_ids):
    return list(range(len(combined_ids)))[-len(val_ids) :]


# Constructing a FastAI DataBlock that uses parsed COCO Dataset from icevision parser. aug_transforms can only be used with_context=True

val_indices = get_val_indices(train_val_record_ids, record_ids_val)


def get_image_by_record_id(record_id):
    return get_image_path(combined_record_collection, record_id)


def get_mask_by_record_id(record_id):
    return record_to_mask(combined_record_collection, record_id)


batch_transfms = [*aug_transforms(flip_vert=True, max_warp=0.1, size=size)]
coco_seg_dblock = DataBlock(
    blocks=(
        ImageBlock,
        MaskBlock(codes=data.class_list),
    ),  # ImageBlock is RGB by default, uses PIL
    get_x=get_image_by_record_id,
    splitter=IndexSplitter(val_indices),
    get_y=get_mask_by_record_id,
    batch_tfms=batch_transfms,
    n_inp=1,
)


dls = coco_seg_dblock.dataloaders(source=train_val_record_ids, batch_size=bs)

# Fastai2 Trainer

dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S")
experiment_dir = Path(f"{mount_path}/experiments/cv2/" + timestampStr + "_fastai_unet/")
experiment_dir.mkdir(exist_ok=True)
print(experiment_dir)

archs = {18: resnet18, 34: resnet34, 50: resnet50}

cbs = [
    TensorBoardCallback(projector=False, trace_model=False),
    SaveModelCallback(monitor="valid_loss", with_opt=True),
    EarlyStoppingCallback(monitor="valid_loss", min_delta=0.005, patience=10),
]

learner = unet_learner(
    dls,
    archs[arch],
    metrics=[DiceMulti, Dice],
    model_dir=experiment_dir,
    n_out=7,
    cbs=cbs,
)  # cbs=cbs# SaveModelCallback saves model when there is improvement

print("size", size)
print("batch size", bs)
print("arch", arch)
print("n chips", n)
print("epochs (with early stopping and patience=10):", epochs)

learner.fine_tune(epochs, 1e-4, freeze_epochs=1)  # cbs=cbs

evaluation.get_cm_for_learner(dls, learner, mount_path)

validation = learner.validate()

save_template = f"test_{bs}_{arch}_{size}_{round(validation[1],3)}_{epochs}.pt"

(
    state_dict_pth,
    tracing_model_gpu_pth,
    tracing_model_cpu_pth,
) = save_fastai_model_state_dict_and_tracing(
    learner, dls, save_template, experiment_dir
)
