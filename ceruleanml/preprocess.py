import pandas as pd
from icevision.data import SingleSplitSplitter
from icevision.parsers import COCOMaskParser

from ceruleanml import load_negative_tiles
from ceruleanml.coco_load_fastai import record_collection_to_record_ids
from ceruleanml.coco_stats import (
    assemble_record_label_lists,
    get_all_record_area_lists_for_class,
    ignore_low_area_records,
)


def get_area_df(coco_json_path, tiled_images_folder):
    parser = COCOMaskParser(
        annotations_filepath=coco_json_path,
        img_dir=tiled_images_folder,
    )
    positive_records = parser.parse(autofix=False, data_splitter=SingleSplitSplitter())[
        0
    ]

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
    coco_json_path, tiled_images_folder, area_thresh, negative_sample_count
):
    parser = COCOMaskParser(
        annotations_filepath=coco_json_path,
        img_dir=tiled_images_folder,
    )
    positive_records = parser.parse(autofix=False, data_splitter=SingleSplitSplitter())[
        0
    ]

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
