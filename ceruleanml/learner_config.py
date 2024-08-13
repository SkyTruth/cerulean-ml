from typing import Dict

from fastai.vision.all import RandomResizedCropGPU , aug_transforms, RandomErasing
from icevision import tfms  # , models

from ceruleanml import coco_load_fastai, data, preprocess

# from torchvision.ops import MultiScaleRoIAlign


model_type = "resnet18"
aux_layers = ["VV"]  # , "INFRA", "VESSEL"]
num_workers = 8  # based on processor, but I don't know how to calculate...

# Note: Scenes are cut into memory-friendly tiles at memtile_size, so that the training loop doesn't need to load a whole GRD when you are just going to RRC it
# The RRC is then executed to reduce the actual training data to rrctile_size
# Finally, the val, test, and serverside datasets are precut to rrctile_size, so that 100% of the data can be evaluated
memtile_size = 1024  # setting memtile_size=0 means use full scenes instead of tiling
rrctile_size = 1024  #

run_list = [
    #   [number of expochs, freeze encoder, augs]
    [30, "unfrozen"],
]

negative_sample_count_train = 0
negative_sample_count_val = 0
negative_sample_count_test = 0
negative_sample_count_rrctrained = 0

area_thresh = 100  # XXX maybe run a histogram on this to confirm that we have much more than 100 px normally!

classes_to_remove = [
    "ambiguous",
    # "natural_seep",
]
classes_to_remap: Dict[str, str] = {
    "old_vessel": "recent_vessel",
    "coincident_vessel": "recent_vessel",
    "infra_slick": "recent_vessel",
    "natural_seep": "recent_vessel",
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

# model_type = models.torchvision.mask_rcnn
# backbone = model_type.backbones.resnext101_32x8d_fpn
# model = model_type.model(
#     backbone=backbone(pretrained=True),
#     num_classes=len(classes_to_keep),
#     box_nms_thresh=0.5,
#     mask_roi_pool=MultiScaleRoIAlign(
#         featmap_names=["0", "1", "2", "3"], output_size=14 * 4, sampling_ratio=2
#     ),
# )

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


reduced_resolution_tile_size=512 #final_px XXX This is mocked for now
scale_limit=0.05
rotate_limit=10
border_mode=0  # cv2.BORDER_CONSTANT, use pad_fill_value
pad_fill_value=[0, 0, 0]  # no_value
mask_value=0
interpolation=0  # cv2.INTER_NEAREST
r_shift_limit=10  # SAR Imagery
g_shift_limit=0  # Infrastructure Vicinity
b_shift_limit=0  # Vessel Density

aug_params_stage1= {
    # "do_flip": True,
    # "flip_vert": True,
    # "max_rotate": rotate_limit,
    # "min_zoom": 1.0 - scale_limit,
    # "max_zoom": 1.0 + scale_limit,
    # "max_lighting": 0,
    # "max_warp": 0.0,
    # "pad_mode": "reflection",
    # "batch": True,
    "do_random_erasing": True,
    "reduced_resolution_tile_size": 256
}


# do_random_erasing = False

aug_schedule = [aug_params_stage1] #stage2, stage3, etc..

if len(aug_schedule) != len(run_list):
    raise("Augmentation Stages does match with Run List")

for i,run in enumerate(run_list):
    run.append(aug_schedule[i])

final_px = aug_schedule[-1]['reduced_resolution_tile_size']

def get_tfms(
    augs
    # do_random_erasing=do_random_erasing,
    # memtile_size=memtile_size,
    # rrctile_size=rrctile_size,

    # reduced_resolution_tile_size=final_px,
    # scale_limit=0.05,
    # rotate_limit=10,
    # border_mode=0,  # cv2.BORDER_CONSTANT, use pad_fill_value
    # pad_fill_value=[0, 0, 0],  # no_value
    # mask_value=0,
    # interpolation=0,  # cv2.INTER_NEAREST
    # r_shift_limit=10,  # SAR Imagery
    # g_shift_limit=0,  # Infrastructure Vicinity
    # b_shift_limit=0,  # Vessel Density
):
    if "mask_rcnn" in model_type:
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
    elif "resnet" in model_type or "convnext" in model_type:
        rrc_crop_area_proportion = (rrctile_size / memtile_size) ** 2
        train_tfms = [
            *aug_transforms(
                mult = augs.get('mult', 1.0),  # Multiplication applying to `max_rotate`, `max_lighting`, `max_warp`
                do_flip = augs.get('do_flip', False),  # Random flipping
                flip_vert = augs.get('flip_vert', False),  # Flip vertically
                max_rotate = augs.get('max_rotate', 0),  # Maximum degree of rotation
                min_zoom = augs.get('min_zoom', 1),  # Minimum zoom 
                max_zoom = augs.get('max_zoom', 1),  # Maximum zoom 
                max_lighting = augs.get('max_lighting', 0),  # Maximum scale of changing brightness 
                max_warp = augs.get('max_warp', 0),  # Maximum value of changing warp per
                p_affine = augs.get('p_affinet', 0),  # Probability of applying affine transformation
                p_lighting = augs.get('p_lighting', 0),  # Probability of changing brightness and contrast 
                xtra_tfms = augs.get('xtra_tfms', None),  # Custom Transformations
                size = augs.get('size', None),  # Output size, duplicated if one value is specified
                mode = augs.get('mode', 'bilinear'),  # PyTorch `F.grid_sample` interpolation
                pad_mode = augs.get('pad_mode', "reflection"),  # A `PadMode`
                align_corners = augs.get('align_corners', True),  # PyTorch `F.grid_sample` align_corners
                batch = augs.get('batch', False),  # Apply identical transformation to entire batch
                min_scale = augs.get('min_scale', 1),  # Minimum scale

            ),
            RandomResizedCropGPU(
                size=augs.get("reduced_resolution_tile_size", 512),
                min_scale=rrc_crop_area_proportion,
                max_scale=rrc_crop_area_proportion,
                ratio=(1, 1),
            ),
            RandomErasing(p=.001),
        ]
        
        if augs.get('do_random_erasing', False):
            train_tfms.append(RandomErasing(max_count=3))

        assert (
            rrc_crop_area_proportion == 1
        ), "WARNING: validation dataset is NOT reduced by RandomResizedCropGPU, so you must use a record_collection pregenerated at the smaller crop size! You may then comment out this assertion."
        valid_tfms = []

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
model_name = f"{len(classes_to_keep)}cls_{model_type}_pr{final_px}_px{rrctile_size}_{sum([r[0] for r in run_list])}epochs"
experiment_name = "AUG_ERASING3" + model_name