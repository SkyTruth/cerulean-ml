import pandas as pd
from icevision.core.class_map import ClassMap
from icevision.data import SingleSplitSplitter
from icevision.parsers import COCOMaskParser

from ceruleanml import load_negative_tiles
from ceruleanml.coco_load_fastai import record_collection_to_record_ids
from ceruleanml.coco_stats import (
    assemble_record_label_lists,
    get_all_record_area_lists_for_class,
    ignore_low_area_records,
)

class_map_coco = {
    "background": 0,
    "infra_slick": 1,
    "natural_seep": 2,
    "coincident_vessel": 3,
    "recent_vessel": 4,
    "old_vessel": 5,
    "ambiguous": 6,
}


class CeruleanCOCOMaskParser(COCOMaskParser):
    def __init__(self, remove_list, remap_dict, class_names_to_keep, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_map = ClassMap(class_names_to_keep)
        self.remove_list = remove_list
        self.remap_dict = remap_dict
        for i in remove_list:
            assert i in class_map_coco.keys()
        self.remove_ids = [class_map_coco[key] for key in self.remove_list]
        img_ids_to_remove = []
        print(
            f"Annotations before filtering classes: {len(self.annotations_dict['annotations'])}"
        )
        print(
            f"Images before filtering classes: {len(self.annotations_dict['images'])}"
        )
        new_cats = []
        class_map_coco_inv = dict((v, k) for k, v in class_map_coco.items())
        for cat in self.annotations_dict["categories"]:
            if cat["id"] in self.remove_ids:
                pass
            else:
                if cat["id"] in remap_dict.keys():
                    cat["name"] = class_map_coco_inv[remap_dict[cat["id"]]]
                    cat["id"] = remap_dict[cat["id"]]
                if cat not in new_cats:
                    new_cats.append(cat)
        filtered_anns = []
        for ann in self.annotations_dict["annotations"]:
            if ann["category_id"] in self.remove_ids:
                img_ids_to_remove.append(ann["image_id"])
            else:
                if ann["category_id"] in remap_dict.keys():
                    ann["category_id"] = remap_dict[ann["category_id"]]
                filtered_anns.append(ann)
        filtered_imgs = []
        for img in self.annotations_dict["images"]:
            if img["id"] not in img_ids_to_remove:
                filtered_imgs.append(img)
        self.annotations_dict["categories"] = new_cats
        self.annotations_dict["annotations"] = filtered_anns
        self.annotations_dict["images"] = filtered_imgs
        print(
            f"Annotations after filtering classes: {len(self.annotations_dict['annotations'])}"
        )
        print(f"Images after filtering classes: {len(self.annotations_dict['images'])}")
        # we need to overwrite the category ids so that they are continuous
        # 0 is still for background
        new_remap_dict = {}
        for i in range(len(new_cats)):
            new_remap_dict[new_cats[i]["id"]] = i + 1
            new_cats[i]["id"] = i + 1
        for ann in self.annotations_dict["annotations"]:
            ann["category_id"] = new_remap_dict[ann["category_id"]]


def get_area_df(
    coco_json_path,
    tiled_images_folder,
    class_names=[
        "infra_slick",
        "natural_seep",
        "coincident_vessel",
        "recent_vessel",
        "old_vessel",
        "ambiguous",
    ],
):
    parser = COCOMaskParser(
        annotations_filepath=coco_json_path,
        img_dir=tiled_images_folder,
    )
    positive_records = parser.parse(autofix=False, data_splitter=SingleSplitSplitter())[
        0
    ]

    record_area_label_lists = assemble_record_label_lists(positive_records)
    dfs = []
    for c in class_names:
        positive_area_label_lists = get_all_record_area_lists_for_class(
            record_area_label_lists, c
        )
        dfs.append(
            pd.DataFrame(
                positive_area_label_lists, columns=["record_id", "label", "area"]
            )
        )
    return pd.concat(dfs)


def load_set_record_collection(
    coco_json_path,
    tiled_images_folder,
    area_thresh=10,
    negative_sample_count=0,
    preprocess=False,
    remove_list=["ambiguous", "natural_seep"],
    remap_dict={  # only remaps coincident and old to recent
        3: 4,
        5: 4,
    },
    class_names_to_keep=[
        "background",
        "infra_slick",
        "recent_vessel",
    ],
):
    """load an icevision record collection with preprocessing steps.

    This function loads a record collection from a coco json with optional preprocessing.

    Args:
        coco_json_path (str): Path to coco json
        tiled_images_folder (str): path to img dir
        area_thresh (int, optional): Annotations (individual masks and their metadata) less than this will be removed from the record collection. Defaults to 10.
        negative_sample_count (int, optional): How many negative samples to randomly add. Defaults to 0.
        preprocess (bool, optional): Boolean flag to allow area thresholding and adding negative samples. Defaults to False.
        remove_list (list[str], optional): List of class name sto remove. Empty list will not remove any. Defaults to ["ambiguous", "natural_seep"].
        remap_dict (dict[int:int], optional): Dict mapping class ids to class ids as identified in class_mapping_coco. Useful for aggregating classes. Defaults to {  # only remaps coincident and old to recent 3: 4, 5: 4, }.
        class_names_to_keep (list[str], optional): Str class names as in coco_mlass_mapping that should remain in the end result (the record collection). Used to define the class mapping for all records. Defaults to [ "background", "infra_slick", "recent_vessel", ].

    Returns:
        icevision.data.record_collection.RecordCollection: A record collection with potentially remapped class ids that differ from class_mapping_coco since they need to be ascending starting from 1.
    """
    if remove_list == [] and remap_dict == {}:
        parser = COCOMaskParser(
            annotations_filepath=coco_json_path, img_dir=tiled_images_folder
        )
    else:
        parser = CeruleanCOCOMaskParser(
            annotations_filepath=coco_json_path,
            img_dir=tiled_images_folder,
            remove_list=remove_list,
            remap_dict=remap_dict,
            class_names_to_keep=class_names_to_keep,
        )
    positive_records = parser.parse(autofix=False, data_splitter=SingleSplitSplitter())[
        0
    ]

    if preprocess:

        ignore_low_area_records(positive_records, area_thresh=area_thresh)

        record_ids = record_collection_to_record_ids(positive_records)

        negative_records = load_negative_tiles.parse_negative_tiles(
            data_dir=tiled_images_folder,
            record_ids=record_ids,
            positive_records=positive_records,
            count=negative_sample_count,
        )
        combined_records = positive_records + negative_records
        return combined_records
    else:
        return positive_records
