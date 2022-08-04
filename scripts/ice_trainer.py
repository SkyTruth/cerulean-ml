import torch
import torchvision
from icevision import Dataset, tfms

from ceruleanml import data, preprocess
from ceruleanml.coco_load_fastai import record_collection_to_record_ids

icevision_experiment_dir = (
    "/root/data/experiments/cv2/20_Jul_2022_00_14_15_icevision_maskrcnn"
)
scripted_model = torch.jit.load(
    f"{icevision_experiment_dir}/scripting_cpu_test_28_34_224_58.pt"
)

run_list = [
    [224, 30]
] * 1  # List of tuples, where the tuples are [px size, training time in minutes]
# run_list = [[64, 1]]*1+[[128, 1]]*1+[[224, 1]]*1 +[[512, 1]]*1
init_size = run_list[0][0]
negative_sample_count = 0
negative_sample_count_val = 100
area_thresh = 0


data_path = "/root/"
mount_path = "/root/data"

val_set = "val-with-context-512"
tiled_images_folder_val = "tiled_images"
json_name_val = "instances_TiledCeruleanDatasetV2.json"
coco_json_path_val = f"{mount_path}/partitions/{val_set}/{json_name_val}"
tiled_images_folder_val = f"{mount_path}/partitions/{val_set}/{tiled_images_folder_val}"
remove_list = ["ambiguous", "natural_seep"]
class_names_to_keep = [
    "background",
    "infra_slick",
    "recent_vessel",
]
remap_dict = {  # only remaps coincident and old to recent
    3: 4,
    5: 4,
}

# since we remove ambiguous and natural seep and remap all vessels to 1 and include background
num_classes = 3
class_map = {v: k for k, v in data.class_mapping_coco_inv.items()}
class_ints = list(range(1, len(list(class_map.keys())[:-1]) + 1))

record_collection_with_negative_small_filtered_val = (
    preprocess.load_set_record_collection(
        coco_json_path_val,
        tiled_images_folder_val,
        area_thresh,
        negative_sample_count_val,
        preprocess=True,
        class_names_to_keep=class_names_to_keep,
        remap_dict=remap_dict,
        remove_list=remove_list,
    )
)
record_ids_val = record_collection_to_record_ids(
    record_collection_with_negative_small_filtered_val
)

valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=init_size)])
valid_ds = Dataset(record_collection_with_negative_small_filtered_val, valid_tfms)
valid_ds = Dataset(
    record_collection_with_negative_small_filtered_val[0:3]
    + record_collection_with_negative_small_filtered_val[100:102]
    + record_collection_with_negative_small_filtered_val[-2:],
    valid_tfms,
)
from icevision.metrics.confusion_matrix import SimpleConfusionMatrix
from icevision.metrics.confusion_matrix.confusion_matrix import MatchingPolicy
from icevision.models.checkpoint import model_from_checkpoint

checkpoint_path = "/root/data/experiments/cv2/20_Jul_2022_00_14_15_icevision_maskrcnn/state_dict_test_28_34_224_58.pt"

class_names = ["Background", "Infrastructure", "Recent Vessel"]

checkpoint_and_model = model_from_checkpoint(
    checkpoint_path,
    model_name="torchvision.mask_rcnn",
    backbone_name="resnet34_fpn",
    img_size=224,
    classes=[
        "background",
        "infra_slick",
        "recent_vessel",
    ],
    is_coco=False,
)

model = checkpoint_and_model["model"]
model_type = checkpoint_and_model["model_type"]
backbone = checkpoint_and_model["backbone"]
class_map = checkpoint_and_model["class_map"]
img_size = checkpoint_and_model["img_size"]
model_type, backbone, class_map, img_size

infer_dl = model_type.infer_dl(valid_ds, batch_size=1, shuffle=False)

preds = model_type.predict_from_dl(
    model, infer_dl, keep_images=True, detection_threshold=0.0
)

cm = SimpleConfusionMatrix()
cm.accumulate(preds)
