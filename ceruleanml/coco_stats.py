import os
from pathlib import Path
from typing import List

import icevision
import numpy as np
import pandas as pd
import skimage.io as skio
from skimage import measure
from tqdm import tqdm

from ceruleanml import data

class_map = {
    "Infrastructure": 1,
    "Natural Seep": 2,
    "Coincident Vessel": 3,
    "Recent Vessel": 4,
    "Old Vessel": 5,
    "Ambiguous": 6,
    "Hard Negatives": 0,
}
class_list = [1, 2, 3, 4, 5, 6]
class_list_string = [
    "Infrastructure",
    "Natural Seep",
    "Coincident Vessel",
    "Recent Vessel",
    "Old Vessel",
    "Ambiguous",
]

class_dict_coco = {
    "infrastructure": "infra_slick",
    "natural": "natural_seep",
    "coincident": "coincident_vessel",
    "recent": "recent_vessel",
    "old": "old_vessel",
    "ambiguous": "ambiguous",
}

class_dict_coco_rgb = {
    "infra_slick": (0, 0, 255),
    "natural_seep": (0, 255, 0),
    "coincident_vessel": (255, 0, 0),
    "recent_vessel": (255, 255, 0),
    "old_vessel": (255, 0, 255),
    "ambiguous": (255, 255, 255),
}

class_map_coco = {
    "background": 0,
    "infra_slick": 1,
    "natural_seep": 2,
    "coincident_vessel": 3,
    "recent_vessel": 4,
    "old_vessel": 5,
    "ambiguous": 6,
}


def sample_stat_lists(arr):
    return np.mean(arr, axis=(0, 1)), np.std(arr, axis=(0, 1))


def all_sample_stat_lists(train_records):
    means = []
    stds = []
    for r in tqdm(train_records):
        p = r.as_dict()["common"]["filepath"]
        arr = skio.imread(p)
        mean_lst, std_lst = sample_stat_lists(arr)
        means.append(mean_lst)
        stds.append(std_lst)
    return np.array(means), np.array(stds)


def minor_axis_length_bbox(bbox):
    """Calculates the minor axis length of the bbox using skimage bbox coordinate order.

    Args:
        bbox (_type_): _description_

    Returns:
        _type_: _description_
    """
    ml1 = bbox[2] - bbox[0]
    ml2 = bbox[3] - bbox[1]
    ml3 = min(ml1, ml2)
    return ml3


def create_mask_skimage_format(record_dict, i):
    m = record_dict["detection"]["masks"][i]
    return np.squeeze(
        m.to_mask(record_dict["common"]["height"], record_dict["common"]["width"]).data
    )


def get_table(d, i, properties):
    image = create_mask_skimage_format(d, i)
    table = measure.regionprops_table(np.squeeze(image), properties=properties)
    return table


def extract_masks_and_compute_tables(
    record_collection: icevision.data.record_collection.RecordCollection,
    instance_label_type: str,
    properties,
):
    tables = []
    for r in record_collection:
        masks = r.as_dict()["detection"]["masks"]
        labels = r.as_dict()["detection"]["labels"]
        for i in range(len(masks)):
            if labels[i] == instance_label_type:
                table = get_table(r.as_dict(), i, properties)
                tables.append(table)
    return tables


def region_props_for_instance_type(
    record_collection: icevision.data.record_collection.RecordCollection,
    instance_label_type: str,
):
    """Calculates the region props for a given type of instance label.

    Args:
        record_collection (icevision.data.record_collection.RecordCollection): A record collection containing instance masks.
        instance_label_type (str): Can be "infra_slick", "natural_seep", "coincident_vessel", "recent_vessel", "old_vessel", "ambiguous"

    Returns:
        _type_: A pandas dataframe of statistics for the instance type.
    """

    img_names = [d.as_dict()["common"]["filepath"] for d in record_collection]

    properties = ["area", "bbox_area", "major_axis_length", "bbox"]

    tables = extract_masks_and_compute_tables(
        record_collection, instance_label_type, properties
    )

    tables = [pd.DataFrame(table) for table in tables]
    for img_name, table in zip(img_names, tables):
        table["img_name"] = img_name
        table["axis_minor_length_bbox"] = minor_axis_length_bbox(
            [
                table["bbox-0"].values,
                table["bbox-1"].values,
                table["bbox-2"].values,
                table["bbox-3"].values,
            ]
        )
        table["category"] = instance_label_type
    if all(v is None for v in tables):
        print(
            f"No categories found to calculate stats for class {instance_label_type}."
        )
        return None
    else:
        rprops_table = pd.concat(tables, axis=0)
        return rprops_table


def get_table_whole_image(path, properties, instance_type):
    image = skio.imread(path)
    r, g, b = class_dict_coco_rgb[instance_type]
    if len(image.shape) == 2 and instance_type == "ambiguous":
        pass
    elif len(image.shape) == 2 and instance_type != "ambiguous":
        return None
    else:
        image = data.rgbalpha_to_binary(image, r, g, b) * 1
    table = measure.regionprops_table(np.squeeze(image), properties=properties)
    table["img_name"] = str(path)
    table["category"] = instance_type
    table["axis_minor_length_bbox"] = minor_axis_length_bbox(
        [
            table["bbox-0"],
            table["bbox-1"],
            table["bbox-2"],
            table["bbox-3"],
        ]
    )
    return table


def extract_masks_and_compute_tables_whole_image(
    record_collection,
    instance_label_type: str,
    properties,
):
    tables = []
    for d in record_collection:
        pths = list(
            Path(os.path.dirname(d.as_dict()["common"]["filepath"])).glob("*_*")
        )
        pths = [
            pth
            for pth in pths
            if "S1A" not in os.path.basename(pth)
            and "S1B" not in os.path.basename(pth)
            and ".DS" not in os.path.basename(pth)
            and ".ipynb" not in os.path.basename(pth)
        ]  # some folders incorrectly contain files names S1A... or S1B...
        for pth in pths:

            if (
                len(os.path.basename(str(pth)).split("_")) != 3
                and len(os.path.basename(str(pth)).split("_")) != 2
            ):
                raise ValueError(f"{os.path.basename(str(pth))}")
            label_type_and_extra_lst = os.path.basename(str(pth)).split("_")
            label_type = []
            for lstr in label_type_and_extra_lst:
                if lstr in class_dict_coco.keys():
                    label_type.append(class_dict_coco[lstr])
            if len(label_type) != 1:
                print("unexpected path, no keywords", label_type_and_extra_lst)
                raise ValueError(f"{label_type}")
            if instance_label_type in label_type:
                table = get_table_whole_image(pth, properties, instance_label_type)
                tables.append(pd.DataFrame(table))
    return tables


def region_props_for_instance_type_whole_image(
    record_collection: icevision.data.record_collection.RecordCollection,
    instance_label_type: str,
):
    """Calculates the region props for a given type of instance label.

    Args:
        record_collection (icevision.data.record_collection.RecordCollection): A record collection containing instance masks.
        instance_label_type (str): Can be "infra_slick", "natural_seep", "coincident_vessel", "recent_vessel", "old_vessel", "ambiguous"

    Returns:
        _type_: A pandas dataframe of statistics for the instance type.
    """

    properties = ["area", "bbox_area", "major_axis_length", "bbox"]

    tables = extract_masks_and_compute_tables_whole_image(
        record_collection, instance_label_type, properties
    )

    if all(v is None for v in tables):
        print(
            f"No categories found to calculate stats for class {instance_label_type}."
        )
        return None
    else:
        rprops_table = pd.concat(tables, axis=0)
        return rprops_table


def get_record_area_label_list(
    positive_train_record: icevision.data.record_collection.RecordCollection,
):
    record_area_label_list = []
    positive_train_record = positive_train_record.as_dict()
    for i in range(len(positive_train_record["detection"]["areas"])):
        record_area_label = (
            positive_train_record["common"]["record_id"],
            positive_train_record["detection"]["labels"][i],
            positive_train_record["detection"]["areas"][i],
        )
        # print(record_area_label)
        record_area_label_list.append(record_area_label)
    return record_area_label_list


def assemble_record_label_lists(
    record_collection: icevision.data.record_collection.RecordCollection,
):
    area_label_lists = []
    for i in range(len(record_collection)):
        area_label_lists.extend(get_record_area_label_list(record_collection[i]))
    return area_label_lists


def get_all_record_area_lists_for_class(record_area_label_lists: List, class_name: str):
    class_names = [
        "infra_slick",
        "natural_seep",
        "coincident_vessel",
        "recent_vessel",
        "old_vessel",
        "ambiguous",
    ]
    assert class_name in class_names
    rs = []
    for r in record_area_label_lists:
        if r[1] == class_name:
            rs.append(r)
    return rs


def ignore_record_by_area(
    record: icevision.data.record_collection.BaseRecord, area_thresh: int
):
    record_d = record.as_dict()
    for i in range(len(record_d["detection"]["areas"])):
        if record_d["detection"]["areas"][i] < area_thresh:
            record_d["detection"]["areas"].pop(i)
            record_d["detection"]["labels"].pop(i)
            record_d["detection"]["label_ids"].pop(i)
            record_d["detection"]["bboxes"].pop(i)
            record_d["detection"]["masks"].pop(i)
            record_d["detection"]["iscrowds"].pop(i)
        return record_d


def ignore_low_area_records(
    record_collection: icevision.data.record_collection.RecordCollection,
    area_thresh: int,
):
    for record in tqdm(record_collection):
        ignore_record_by_area(record, area_thresh)

def remap_records_class(
    record_collection: icevision.data.record_collection.RecordCollection,
    remap_dict: dict,
):
    """
    remap_dict keys/values: "infra_slick", "natural_seep", "coincident_vessel", "recent_vessel", "old_vessel", "ambiguous", None
    Deletes class if value = None
    """
    for record in tqdm(record_collection):
        record_d = record.as_dict()
        for i, label in enumerate(record_d["detection"]["labels"]):
            if label in remap_dict:
                if remap_dict[label] == None:
                    [record_d["detection"][key].pop(i) for key, value in record_d["detection"].items() if value is not None]
                else:
                    record_d["detection"]["labels"][i] = remap_dict[label]
                    record_d["detection"]["label_ids"][i] = class_map_coco[remap_dict[label]]
