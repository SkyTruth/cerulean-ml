from typing import Dict

from icevision import models, tfms
from torchvision.ops import MultiScaleRoIAlign

from ceruleanml import coco_load_fastai, data, preprocess

# Note: Scenes are cut into memory-friendly tiles at memtile_size, so that the training loop doesn't need to load a whole GRD when you are just going to RRC it
# The RRC is then executed to reduce the actual training data to rrctile_size
# Finally, the val, test, and serverside datasets are precut to rrctile_size, so that 100% of the data can be evaluated
memtile_size = 1024  # setting memtile_size=0 means use full scenes instead of tiling
rrctile_size = 1024  #
run_list = [
    [512, 80],
    # [416, 60],
]  # List of tuples, where the tuples are [px size, training time in minutes]
final_px = run_list[-1][0]

negative_sample_count_train = 100
negative_sample_count_val = 0
negative_sample_count_test = 0
negative_sample_count_rrctrained = 0

area_thresh = 100  # XXX maybe run a histogram on this to confirm that we have much more than 100 px normally!

classes_to_remove = [
    "ambiguous",
    # "natural_seep",
]
classes_to_remap: Dict[str, str] = {
    # "old_vessel": "recent_vessel",
    # "coincident_vessel": "recent_vessel",
}

classes_to_keep = [
    c
    for c in data.class_list
    if c not in classes_to_remove + list(classes_to_remap.keys())
]

thresholds = {
    "pixel_nms_thresh": 0.4,  # prediction vs itself, pixels
    "bbox_score_thresh": 0.2,  # prediction vs score, bbox
    "poly_score_thresh": 0.2,  # prediction vs score, polygon
    "pixel_score_thresh": 0.2,  # prediction vs score, pixels
    "groundtruth_dice_thresh": 0.0,  # prediction vs ground truth, theshold
}

num_workers = 8  # based on processor, but I don't know how to calculate...
model_type = models.torchvision.mask_rcnn
backbone = model_type.backbones.resnext101_32x8d_fpn
model = model_type.model(
    backbone=backbone(pretrained=True),
    num_classes=len(classes_to_keep),
    box_nms_thresh=0.5,
    mask_roi_pool=MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=14 * 4, sampling_ratio=2
    ),
)

# Regularization
wd = 0.01


# Ablation studies for aux channels
def triplicate(img, **params):
    img[..., :] = img[..., 0:1]
    return img


def sat_mask(img, **params):
    img[..., :] = img[..., 0:1]
    img[..., 2] = img[..., 2] != 0
    return img


def vessel_traffic(img, **params):
    img[..., 1] = img[..., 0]
    return img


def infra_distance(img, **params):
    img[..., 2] = img[..., 0]
    return img


def no_op(img, **params):
    return img


def get_tfms(
    memtile_size=memtile_size,
    rrctile_size=rrctile_size,
    reduced_resolution_tile_size=final_px,
    scale_limit=0.05,
    rotate_limit=10,
    border_mode=0,  # cv2.BORDER_CONSTANT, use pad_fill_value
    pad_fill_value=[0, 0, 0],  # no_value
    mask_value=0,
    interpolation=0,  # cv2.INTER_NEAREST
    r_shift_limit=10,  # SAR Imagery
    g_shift_limit=0,  # Infrastructure Vicinity
    b_shift_limit=0,  # Vessel Density
):
    train_tfms = tfms.A.Adapter(
        [
            tfms.A.Flip(
                p=0.5,
            ),
            tfms.A.Affine(
                p=1,
                scale=(1 - scale_limit, 1 + scale_limit),
                rotate=[-rotate_limit, rotate_limit],
                interpolation=interpolation,
                mode=border_mode,
                cval=pad_fill_value,
                cval_mask=mask_value,
                fit_output=True,
            ),
            tfms.A.RandomSizedCrop(
                p=1,
                min_max_height=[rrctile_size, rrctile_size],
                height=reduced_resolution_tile_size,
                width=reduced_resolution_tile_size,
                w2h_ratio=1,
                interpolation=interpolation,
            ),
            tfms.A.RGBShift(
                p=1,
                r_shift_limit=r_shift_limit,
                g_shift_limit=g_shift_limit,
                b_shift_limit=b_shift_limit,
            ),
            tfms.A.Lambda(p=1, image=no_op),
        ]
    )
    valid_tfms = tfms.A.Adapter(
        [
            tfms.A.RandomSizedCrop(
                p=1,
                min_max_height=[rrctile_size, rrctile_size],
                height=reduced_resolution_tile_size,
                width=reduced_resolution_tile_size,
                w2h_ratio=1,
                interpolation=interpolation,
            ),
            tfms.A.Lambda(p=1, image=no_op),
        ]
    )

    return [train_tfms, valid_tfms]


# Datasets
mount_path = "/root"

# Parsing COCO Dataset with Icevision
json_name = "instances_TiledCeruleanDatasetV2.json"

train_set = f"train_tiles_context_{memtile_size}"
coco_json_path_train = f"{mount_path}/partitions/{train_set}/{json_name}"
tiled_images_folder_train = f"{mount_path}/partitions/{train_set}/tiled_images"

val_set = f"val_tiles_context_{rrctile_size}"
coco_json_path_val = f"{mount_path}/partitions/{val_set}/{json_name}"
tiled_images_folder_val = f"{mount_path}/partitions/{val_set}/tiled_images"

test_set = f"test_tiles_context_{rrctile_size}"
coco_json_path_test = f"{mount_path}/partitions/{test_set}/{json_name}"
tiled_images_folder_test = f"{mount_path}/partitions/{test_set}/tiled_images"

rrctrained_set = f"train_tiles_context_{rrctile_size}"
coco_json_path_rrctrained = f"{mount_path}/partitions/{rrctrained_set}/{json_name}"
tiled_images_folder_rrctrained = (
    f"{mount_path}/partitions/{rrctrained_set}/tiled_images"
)

record_collection_train = preprocess.load_set_record_collection(
    coco_json_path_train,
    tiled_images_folder_train,
    area_thresh,
    negative_sample_count_train,
    preprocess=True,
    classes_to_remap=classes_to_remap,
    classes_to_remove=classes_to_remove,
    classes_to_keep=classes_to_keep,
)

record_collection_val = preprocess.load_set_record_collection(
    coco_json_path_val,
    tiled_images_folder_val,
    area_thresh,
    negative_sample_count_val,
    preprocess=True,
    classes_to_remap=classes_to_remap,
    classes_to_remove=classes_to_remove,
    classes_to_keep=classes_to_keep,
)
for record in record_collection_val:
    record.set_record_id(
        record.record_id + len(record_collection_train)
    )  # Increment the record ID to avoid clashes
    record.record_id += len(
        record_collection_train
    )  # Increment the record ID to avoid clashes
    record.common.record_id += len(
        record_collection_train
    )  # Increment the record ID to avoid clashes

record_collection_test = preprocess.load_set_record_collection(
    coco_json_path_test,
    tiled_images_folder_test,
    area_thresh,
    negative_sample_count_test,
    preprocess=True,
    classes_to_remap=classes_to_remap,
    classes_to_remove=classes_to_remove,
    classes_to_keep=classes_to_keep,
)
for record in record_collection_test:
    record.set_record_id(
        record.record_id + len(record_collection_train) + len(record_collection_val)
    )  # Increment the record ID to avoid clashes
    record.record_id += len(record_collection_train) + len(
        record_collection_val
    )  # Increment the record ID to avoid clashes
    record.common.record_id += len(
        record_collection_train
    )  # Increment the record ID to avoid clashes

# record_collection_rrctrained = preprocess.load_set_record_collection(
#     coco_json_path_rrctrained,
#     tiled_images_folder_rrctrained,
#     area_thresh,
#     negative_sample_count_rrctrained,
#     preprocess=True,
#     classes_to_remap=classes_to_remap,
#     classes_to_remove=classes_to_remove,
#     classes_to_keep=classes_to_keep,
# )

# Confirm that train and val are mutually exclusive collections
record_ids_train = coco_load_fastai.record_collection_to_record_ids(
    record_collection_train
)
record_ids_val = coco_load_fastai.record_collection_to_record_ids(record_collection_val)
record_ids_test = coco_load_fastai.record_collection_to_record_ids(
    record_collection_test
)

# Create name for model based on parameters above
model_name = f"{len(classes_to_keep)}cls_rnxt101_pr{run_list[-1][0]}_px{rrctile_size}_{sum([r[1] for r in run_list])}min"
