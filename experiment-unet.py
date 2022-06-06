import os

# from fastai.callback.fp16 import MixedPrecision
# from fastai.callback.tensorboard import TensorBoardCallback
from datetime import datetime
from pathlib import Path

import numpy as np
import skimage.io as skio
from fastai.data.block import DataBlock
from fastai.data.transforms import IndexSplitter, Normalize
from fastai.metrics import DiceMulti

# from fastai.vision.augment import aug_transforms
from fastai.vision.data import ImageBlock, MaskBlock
from fastai.vision.learner import unet_learner
from omegaconf import DictConfig
from torchvision.models import resnet18, resnet34, resnet50

import hydra
from ceruleanml import data, evaluation, preprocess
from ceruleanml.coco_load_fastai import (
    get_image_path,
    record_collection_to_record_ids,
    record_to_mask,
)
from ceruleanml.inference import save_fastai_model_state_dict_and_tracing


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

    class_map = {v: k for k, v in data.class_mapping_coco_inv.items()}
    class_ints = list(range(1, len(list(class_map.keys())[:-1]) + 1))
    mount_path = "../.."
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
    tiled_images_folder_val = (
        f"{mount_path}/partitions/{val_set}/{tiled_images_folder_val}"
    )

    class_map = {v: k for k, v in data.class_mapping_coco_inv.items()}
    class_ints = list(range(1, len(list(class_map.keys())[:-1]) + 1))
    negative_sample_count = 100
    negative_sample_count_val = 50
    mean = [60.73, 190.3, 4.3598]
    std = [16.099, 17.846, 9.603]
    area_thresh = 10
    record_collection_with_negative_small_filtered_train = (
        preprocess.load_set_record_collection(
            coco_json_path_train,
            tiled_images_folder_train,
            area_thresh,
            negative_sample_count,
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
        )
    )
    record_ids_val = record_collection_to_record_ids(
        record_collection_with_negative_small_filtered_val
    )
    # check for 0 overlap between train and val
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

    # size = 64  # Progressive resizing could happen here
    # batch_transfms = [aug_transforms(flip_vert=True, max_warp=0.1, size=size), Normalize.from_stats(mean, std)]

    val_indices = get_val_indices(train_val_record_ids, record_ids_val)

    def get_image_by_record_id(record_id):
        return get_image_path(combined_record_collection, record_id)

    def get_mask_by_record_id(record_id):
        return record_to_mask(combined_record_collection, record_id)

    coco_seg_dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=class_ints)),
        get_x=get_image_by_record_id,
        splitter=IndexSplitter(val_indices),
        get_y=get_mask_by_record_id,
        batch_tfms=[Normalize.from_stats(mean, std)],
        n_inp=1,
    )

    dls = coco_seg_dblock.dataloaders(source=train_val_record_ids, batch_size=6, n=16)

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S")
    experiment_dir = Path(
        f"{mount_path}/experiments/cv2/" + timestampStr + "_fastai_unet/"
    )
    experiment_dir.mkdir(exist_ok=True)

    arch = 18
    archs = {18: resnet18, 34: resnet34, 50: resnet50}

    learner = unet_learner(
        dls,
        archs[arch],
        metrics=[DiceMulti(axis=1)],
        model_dir=experiment_dir,
        n_out=7,
    )
    learner.fine_tune(2, 2e-4, freeze_epochs=1)

    size = 512
    # savename = f'test_6batch_{arch}_{size}_{round(validation[1],3)}.pt'
    savename = f"test_6batch_{arch}_{size}.pt"
    # TODO these files should be logged and saved in a standard dir on gcp
    (
        state_dict_pth,
        tracing_model_gpu_pth,
        tracing_model_cpu_pth,
    ) = save_fastai_model_state_dict_and_tracing(learner, dls, savename, experiment_dir)

    pred_arrs = []
    val_arrs = []
    for v, i in zip(range(len(record_ids_val[0:10])), record_ids_val[0:10]):
        val_record = record_collection_with_negative_small_filtered_val[v]
        v_masks = (
            val_record.detection.masks[0]
            .to_mask(val_record.common.height, val_record.common.width)
            .data
        )
        p = get_image_path(record_collection_with_negative_small_filtered_val, i)
        arr = skio.imread(p)
        # necessary for 1 channel input since fastai uses PIL during predict
        class_pred = learner.predict(np.squeeze(arr))
        class_pred = class_pred[0].cpu().detach().numpy()
        pred_arrs.append(class_pred)
        val_arrs.append(v_masks)

    cm, f1 = evaluation.cm_f1(
        val_arrs, pred_arrs, 6, mount_path
    )  # todo add normalize false

    return cm, f1


if __name__ == "__main__":
    train()
