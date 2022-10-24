from icevision import models, tfms

from ceruleanml import coco_load_fastai, data, preprocess

super_tile_size = 2048
target_tile_size = 1024  #
run_list = [
    [512, 1 * 60],
]  # List of tuples, where the tuples are [px size, training time in minutes]

negative_sample_count_train = 0
negative_sample_count_val = 0
negative_sample_count_test = 0

area_thresh = 0  # XXX maybe run a histogram on this to confirm that we have much more than 100 px normally!

classes_to_remove = [
    "ambiguous",
    # "natural_seep",
]
classes_to_remap = {
    "old_vessel": "recent_vessel",
    "coincident_vessel": "recent_vessel",
}

classes_to_keep = [
    c
    for c in data.class_list
    if c not in classes_to_remove + list(classes_to_remap.keys())
]


num_workers = 8  # based on processor, but I don't know how to calculate...
model_type = models.torchvision.mask_rcnn
backbone = model_type.backbones.resnet50_fpn
model = model_type.model(
    backbone=backbone(pretrained=True),
    num_classes=len(classes_to_keep),
    box_nms_thresh=0.5,
)

# Regularization
wd = 0.1


def get_tfms(
    super_tile_size=super_tile_size,
    target_tile_size=target_tile_size,
    reduced_resolution_tile_size=run_list[-1][0],
    scale_limit=0.05,
    rotate_limit=180,
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
                scale=(
                    1 - scale_limit,
                    1 + scale_limit,
                ),  # Note cannot enforce keep_ratio=True, so x and y will be scaled independently!
                rotate=[-rotate_limit, rotate_limit],
                interpolation=interpolation,
                mode=border_mode,
                cval=pad_fill_value,
                cval_mask=mask_value,
                fit_output=True,
            ),
            tfms.A.RandomResizedCrop(
                p=1,
                height=reduced_resolution_tile_size,
                width=reduced_resolution_tile_size,
                scale=(
                    (target_tile_size / super_tile_size) ** 2,
                    (target_tile_size / super_tile_size) ** 2,
                ),
                ratio=(1, 1),  # Never change perspective ratio
                interpolation=interpolation,
            ),
            tfms.A.RGBShift(
                p=1,
                r_shift_limit=r_shift_limit,
                g_shift_limit=g_shift_limit,
                b_shift_limit=b_shift_limit,
            ),
        ]
    )
    valid_tfms = tfms.A.Adapter(
        [
            tfms.A.RandomResizedCrop(
                height=reduced_resolution_tile_size,
                width=reduced_resolution_tile_size,
                scale=(
                    (target_tile_size / super_tile_size) ** 2,
                    (target_tile_size / super_tile_size) ** 2,
                ),
                ratio=(1, 1),  # Never change perspective ratio
                interpolation=interpolation,
            ),
        ]
    )

    return [train_tfms, valid_tfms]


# Datasets
mount_path = "/root"

# Parsing COCO Dataset with Icevision
tile_size = f"{super_tile_size}"
tiled_images_folder = "tiled_images"
json_name = "instances_TiledCeruleanDatasetV2.json"

train_set = f"train_tiles_context_{tile_size}"
coco_json_path_train = f"{mount_path}/partitions/{train_set}/{json_name}"
tiled_images_folder_train = f"{mount_path}/partitions/{train_set}/{tiled_images_folder}"

val_set = f"val_tiles_context_{tile_size}"
coco_json_path_val = f"{mount_path}/partitions/{val_set}/{json_name}"
tiled_images_folder_val = f"{mount_path}/partitions/{val_set}/{tiled_images_folder}"

test_set = f"test_tiles_context_{tile_size}"
coco_json_path_test = f"{mount_path}/partitions/{test_set}/{json_name}"
tiled_images_folder_test = f"{mount_path}/partitions/{test_set}/{tiled_images_folder}"

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

# Confirm that train and val are mutually exclusive collections
record_ids_train = coco_load_fastai.record_collection_to_record_ids(
    record_collection_train
)
record_ids_val = coco_load_fastai.record_collection_to_record_ids(record_collection_val)
record_ids_test = coco_load_fastai.record_collection_to_record_ids(
    record_collection_test
)
print(
    f"{len(set(record_ids_train + record_ids_val + record_ids_test))} =? {len(record_ids_train + record_ids_val + record_ids_test)}"
)
# assert len(set(record_ids_train + record_ids_val + record_ids_test)) == len(record_ids_train + record_ids_val + record_ids_test)

# Create name for model based on parameters above
model_name = f"{len(classes_to_keep)}cls_rn50_pr{run_list[-1][0]}_px{tile_size}_{sum([r[1] for r in run_list])}min"
