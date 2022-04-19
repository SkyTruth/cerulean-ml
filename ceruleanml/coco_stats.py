import icevision
import numpy as np
import pandas as pd
from skimage import measure

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
    ml1 = bbox[2] - bbox[0]
    ml2 = bbox[3] - bbox[1]
    ml3 = min(ml1, ml2)
    return ml3


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

    def create_mask_skimage_format(d, i):
        m = d["detection"]["masks"][i]
        return np.squeeze(m.to_mask(d["common"]["height"], d["common"]["width"]).data)

    mask_arrays = []
    for r in record_collection:
        masks = r.as_dict()["detection"]["masks"]
        labels = r.as_dict()["detection"]["labels"]
        for i in range(len(masks)):
            if labels[i] == instance_label_type:
                mask_arrays.append(create_mask_skimage_format(r.as_dict(), i))

    img_names = [d.as_dict()["common"]["filepath"] for d in record_collection]

    properties = ["area", "bbox_area", "major_axis_length", "bbox"]

    tables = [
        measure.regionprops_table(np.squeeze(image), properties=properties)
        for image in mask_arrays
    ]
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
