from typing import List

import icevision
import numpy as np
import pandas as pd

# os.chdir(os.path.dirname(os.path.abspath(__file__)))


def group_record_ids_by_sample(
    record_collection: icevision.data.record_collection.RecordCollection,
):
    # check that each record only has one sparse mask
    for i in range(len(record_collection)):
        assert len(record_collection[0].as_dict()["detection"]["masks"]) == 1
    filepaths = []
    record_ids = []
    for r in record_collection:
        filepaths.append(r.as_dict()["common"]["filepath"])
        record_ids.append(r.as_dict()["common"]["record_id"])
    df = pd.DataFrame({"filepaths": filepaths, "record_ids": record_ids}).reset_index()
    return df.groupby(["filepaths"])["record_ids"].apply(list)


def get_image_path(record_collection, record_id_list):
    d = record_collection.get_by_record_id(record_id_list[0]).as_dict()
    return d["common"]["filepath"]


def record_to_mask(
    record_collection: icevision.data.record_collection.RecordCollection,
    record_id_list: List,
):
    """Takes a record collection containing coco dataset annotations and
        converts annotations to semantic masks.

    Instance masks are merged together so that instance information is lost
    and the first instance in a list of instances is given priority to be
    included in the final semantic mask output if multiple instances overlap.

    Args:
        record_collection (icevision.data.record_collection.RecordCollection):
            An icevision record collection. Best accessed by converting elements
            to a regular dict. Used to access records (instances) by unique ID.
        record_id_list (List[List]): A list of group ids.  Should reference
            all instances associated with an image tile.

    Returns:
        _type_: A
    """
    if len(record_id_list) == 1:
        d = record_collection.get_by_record_id(record_id_list[0]).as_dict()
        return (
            d["detection"]["masks"][0]
            .to_mask(d["common"]["height"], d["common"]["width"])
            .data.squeeze()
        )
    elif len(record_id_list) > 1:
        arrs = []
        for i in record_id_list:
            d = record_collection.get_by_record_id(i).as_dict()
            arr = (
                d["detection"]["masks"][0]
                .to_mask(d["common"]["height"], d["common"]["width"])
                .data
            )
            arr = (
                arr * d["detection"]["label_ids"][0]
            )  # each record only contains one instance, binary to id
            arrs.append(arr)
        add_mask = (np.concatenate(arrs) > 0).sum(
            axis=0
        ) <= 1  # True where there are no overlapping class ids
        out = arrs[0].copy()  # if there's overlap, we just assign the first class
        return np.add.reduce(arrs, where=add_mask, out=out).squeeze()
