import icevision
import numpy as np


# os.chdir(os.path.dirname(os.path.abspath(__file__)))
def record_collection_to_record_ids(
    record_collection: icevision.data.record_collection.RecordCollection,
):
    # check that each record only has one sparse mask
    record_ids = []
    for r in record_collection:
        record_ids.append(r.as_dict()["common"]["record_id"])
    return record_ids


def get_image_path(record_collection, record_id):
    d = record_collection.get_by_record_id(record_id).as_dict()
    return d["common"]["filepath"]


def record_to_mask(
    record_collection: icevision.data.record_collection.RecordCollection, record_id: int
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
        record_id (int): A record id referencing all instances associated with an
            image tile.

    Returns:
        np.ndarray: A semantic segmentation label array
    """
    d = record_collection.get_by_record_id(record_id).as_dict()
    arrs = []
    for label_index, sparse_mask in enumerate(d["detection"]["masks"]):
        arr = sparse_mask.to_mask(d["common"]["height"], d["common"]["width"]).data
        arr = (
            arr * d["detection"]["label_ids"][label_index]
        )  # each record only contains one instance, binary to id
        arrs.append(arr)
    add_mask = (np.concatenate(arrs) > 0).sum(
        axis=0
    ) <= 1  # True where there are no overlapping class ids
    out = arrs[0].copy()  # if there's overlap, we just assign the first class
    return np.add.reduce(arrs, where=add_mask, out=out).squeeze()
