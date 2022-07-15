import pandas as pd
from icevision.data import SingleSplitSplitter
from icevision.parsers import COCOMaskParser
from icevision.core.class_map import ClassMap
from ceruleanml import load_negative_tiles
from ceruleanml.coco_load_fastai import record_collection_to_record_ids
from ceruleanml.coco_stats import (
    assemble_record_label_lists,
    get_all_record_area_lists_for_class,
    ignore_low_area_records,
    remap_records_class,
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
        self.remove_ids = [class_map_coco[key] for key in self.remove_list]

    def parse_fields(self, o, record, is_new):
        if o["category_id"] in self.remove_ids:
            pass
        elif o["category_id"] in self.remap_dict.keys():
            o["category_id"] = self.remap_dict[o["category_id"]]
            super().parse_fields(o, record, is_new=is_new)
            record.detection.add_masks(self.masks(o))
        else:
            super().parse_fields(o, record, is_new=is_new)
            record.detection.add_masks(self.masks(o))


def get_area_df(coco_json_path, tiled_images_folder):
    parser = COCOMaskParser(
        annotations_filepath=coco_json_path,
        img_dir=tiled_images_folder,
    )
    positive_records = parser.parse(autofix=False, data_splitter=SingleSplitSplitter())[0]

    class_names = [
        "infra_slick",
        "natural_seep",
        "coincident_vessel",
        "recent_vessel",
        "old_vessel",
        "ambiguous",
    ]
    record_area_label_lists = assemble_record_label_lists(positive_records)
    dfs = []
    for c in class_names:
        positive_area_label_lists = get_all_record_area_lists_for_class(record_area_label_lists, c)
        dfs.append(pd.DataFrame(positive_area_label_lists, columns=["record_id", "label", "area"]))
    return pd.concat(dfs)


def load_set_record_collection(
    coco_json_path,
    tiled_images_folder,
    area_thresh=10,
    negative_sample_count=0,
    preprocess=False,
    remap_dict={},
):
    parser = CeruleanCOCOMaskParser(
        annotations_filepath=coco_json_path,
        img_dir=tiled_images_folder,
        remove_list=["ambiguous"],
        remap_dict={},
        class_names_to_keep=[
            "infra_slick",
            "natural_seep",
            "coincident_vessel",
            "recent_vessel",
            "old_vessel",
        ],
    )
    positive_records = parser.parse(autofix=False, data_splitter=SingleSplitSplitter())[0]

    # remap_records_class(positive_records, remap_dict)

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
