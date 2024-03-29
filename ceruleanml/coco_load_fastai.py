import icevision
import numpy as np


# os.chdir(os.path.dirname(os.path.abspath(__file__)))
def record_collection_to_record_ids(
    record_collection: icevision.data.record_collection.RecordCollection,
):
    """Extracts record ids from an icevision record collection.

    Useful for setting up the list of sources for a fastai dataloader.
    As opposed to a more traditional source like a folder path.

    Args:
        record_collection (icevision.data.record_collection.RecordCollection): The collection of
        label data and metadata.

    Returns:
        List[int]: A list of unique record ids in the same order as elements in the record collection.
    """
    return [r.common.record_id for r in record_collection]


def get_image_path(
    record_collection: icevision.data.record_collection.RecordCollection, record_id: int
):
    """Gets img_path for a record_id from the record_collection

    Args:
        record_collection (icevision.data.record_collection.RecordCollection): The collection of
        label data and metadata.
        record_id (int): The unique record id.

    Returns:
        str: Path to the image tile of the record.
    """
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
    if len(d["detection"]["masks"]) > 0:
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
    else:
        return np.zeros((d["common"]["height"], d["common"]["width"]), dtype=np.int8)
