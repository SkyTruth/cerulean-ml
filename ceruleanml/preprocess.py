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
from ceruleanml.data import class_idx_dict, class_list


class CeruleanCOCOMaskParser(COCOMaskParser):
    def __init__(
        self, classes_to_remove, classes_to_remap, held_scenes, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.classes_to_remove = classes_to_remove
        self.classes_to_remap = classes_to_remap
        self.held_scenes = held_scenes
        # Assert that the coco json class list is fully within the data.py class_list
        assert all(
            [
                class_list[coco_dict["id"]] == coco_dict["name"]
                for coco_dict in self.annotations_dict["categories"]
            ]
        )
        # Assert that all the supplied classes are within the data.py class_list
        assert all(
            [
                target_class in class_list
                for target_class in classes_to_remove
                + list(classes_to_remap.keys())
                + list(classes_to_remap.values())
            ]
        )
        # Assert that we are not mapping to a removed class
        assert all(
            [
                target_class not in classes_to_remove
                for target_class in classes_to_remap.values()
            ]
        )

        classes_to_keep = class_list.copy()
        for category in classes_to_remove + list(classes_to_remap.keys()):
            classes_to_keep.remove(category)
        self.class_map = ClassMap(class_list)

        print(
            f"Annotations before filtering classes: {len(self.annotations_dict['annotations'])}"
        )
        print(
            f"Images before filtering classes: {len(self.annotations_dict['images'])}"
        )

        new_cats = [
            cat
            for cat in self.annotations_dict["categories"]
            if cat["name"] in classes_to_keep
        ]
        filtered_anns = []
        img_ids_to_remove = []
        for ann in self.annotations_dict["annotations"]:
            ann_class = class_list[ann["category_id"]]
            if (ann_class in classes_to_remove) or (
                ann["big_image_original_fname"].split(".")[0] in held_scenes
            ):
                # TODO the held_scenes should not be used here--just a temporary hack because the code to split the Datasets did not hold out the test or valid sets from the training list
                img_ids_to_remove.append(ann["image_id"])
            else:
                if ann_class in classes_to_remap:
                    ann["category_id"] = class_idx_dict[classes_to_remap[ann_class]]
                filtered_anns.append(ann)

        filtered_imgs = [
            img
            for img in self.annotations_dict["images"]
            if img["id"] not in img_ids_to_remove
        ]

        self.annotations_dict["categories"] = new_cats
        self.annotations_dict["annotations"] = filtered_anns
        self.annotations_dict["images"] = filtered_imgs

        print(
            f"Annotations after filtering classes: {len(self.annotations_dict['annotations'])}"
        )
        print(f"Images after filtering classes: {len(self.annotations_dict['images'])}")


def get_area_df(
    coco_json_path,
    tiled_images_folder,
    class_names=[],
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
    coco_json_path,  # This is where the annotations_dict lives
    tiled_images_folder,
    area_thresh=10,
    negative_sample_count=0,
    preprocess=False,
    classes_to_remove=[],
    classes_to_remap={},
    held_scenes=[],
):
    """load an icevision record collection with optional preprocessing steps controlled by a flag.

    This function loads a record collection from a coco json with optional preprocessing.

    Args:
        coco_json_path (str): Path to coco json
        tiled_images_folder (str): path to img dir
        area_thresh (int, optional): Annotations (individual masks and their metadata) less than this will be removed from the record collection. Defaults to 10.
        negative_sample_count (int, optional): How many negative samples to randomly add. Defaults to 0.
        preprocess (bool, optional): Boolean flag to allow area thresholding and adding negative samples. Defaults to False.
        classes_to_remove (list[str], optional): List of class name sto remove. Empty list will not remove any. Defaults to ["ambiguous", "natural_seep"].
        classes_to_remap (dict[int:int], optional): Dict mapping class ids to class ids as identified in class_list. Useful for aggregating classes. Defaults to {  # only remaps coincident and old to recent 3: 4, 5: 4, }.

    Returns:
        icevision.data.record_collection.RecordCollection: A record collection.
    """
    parser = CeruleanCOCOMaskParser(
        annotations_filepath=coco_json_path,
        img_dir=tiled_images_folder,
        classes_to_remove=classes_to_remove,
        classes_to_remap=classes_to_remap,
        held_scenes=held_scenes,
    )
    positive_records = parser.parse(autofix=False, data_splitter=SingleSplitSplitter())[
        0
    ]  # single split comes nested so we get the actual record

    if preprocess:
        print(
            "applying preprocessing steps, adding negative samples and filtering low area"
        )
        ignore_low_area_records(positive_records, area_thresh=area_thresh)

        record_ids = record_collection_to_record_ids(positive_records)

        negative_records = load_negative_tiles.parse_negative_tiles(
            data_dir=tiled_images_folder,
            record_ids=record_ids,
            positive_records=positive_records,
            count=negative_sample_count,
            class_names=class_list,
        )
        combined_records = positive_records + negative_records
        return combined_records
    else:
        return positive_records
