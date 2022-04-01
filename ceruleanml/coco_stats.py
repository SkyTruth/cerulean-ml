from skimage import measure
import pandas as pd
import icevision

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


def minor_axis_length_bbox(bbox):
    """Calculates the minor axis length of the bbox using skimage bbox coordinate order.

    Args:
        bbox (_type_): _description_

    Returns:
        _type_: _description_
    """
    ml1 = bbox[3] - bbox[0]
    ml2 = bbox[4] - bbox[1]
    ml3 = min(ml1, ml2)
    return ml3


def region_props_for_instance_type(
    record_collection: icevision.data.record_collection.RecordCollection, instance_label_type: str
):
    """Calculates the region props for a given type of instance label.

    Args:
        record_collection (icevision.data.record_collection.RecordCollection): A record collection containing instance masks.
        instance_label_type (str): Can be "infra_slick", "natural_seep", "coincident_vessel", "recent_vessel", "old_vessel", "ambiguous"

    Returns:
        _type_: A pandas dataframe of statistics for the instance type.
    """
    create_mask_skimage_format = (
        lambda d: d.as_dict()["detection"]["masks"][0]
        .to_mask(d.as_dict()["common"]["height"], d.as_dict()["common"]["width"])
        .data.transpose(1, 2, 0)
    )
    mask_arrays = [
        create_mask_skimage_format(d)
        for d in record_collection
        if d.as_dict()["detection"]["labels"] == [instance_label_type]
    ]

    img_names = [d.as_dict()["common"]["filepath"] for d in record_collection]

    properties = ["area", "area_bbox", "axis_major_length", "bbox"]

    tables = [measure.regionprops_table(image, properties=properties) for image in mask_arrays]
    tables = [pd.DataFrame(table) for table in tables]
    for img_name, table in zip(img_names, tables):
        table["img_name"] = img_name
        table["axis_minor_length_bbox"] = minor_axis_length_bbox(table["bbox"])
    rprops_table = pd.concat(tables, axis=0)

    return rprops_table
