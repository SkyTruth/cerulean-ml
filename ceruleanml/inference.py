import geojson
import numpy as np
import torch
from rasterio.features import shapes
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from shapely.geometry import MultiPolygon, shape
from tqdm import tqdm


def save_fastai_model_state_dict_and_tracing(learner, dls, savename, experiment_dir):
    sd = learner.model.state_dict()
    torch.save(
        sd, f"{experiment_dir}/state_dict_{savename}"
    )  # saves state_dict for loading with fastai
    x, _ = dls.one_batch()
    learner.model.cuda()
    learner.model.eval()
    torch.jit.save(
        torch.jit.trace(learner.model, x), f"{experiment_dir}/tracing_gpu_{savename}"
    )
    learner.model.to("cpu")
    torch.jit.save(
        torch.jit.trace(learner.model, x.to("cpu")),
        f"{experiment_dir}/tracing_cpu_{savename}",
    )
    print(f"{experiment_dir}/tracing_gpu_{savename}")
    print(f"{experiment_dir}/tracing_cpu_{savename}")
    print(f"{experiment_dir}/state_dict_{savename}")
    return (
        f"{experiment_dir}/state_dict_{savename}",
        f"{experiment_dir}/tracing_gpu_{savename}",
        f"{experiment_dir}/tracing_cpu_{savename}",
    )


def save_icevision_model_state_dict_and_tracing(learner, savename, experiment_dir):
    sd = learner.model.state_dict()
    torch.save(
        sd, f"{experiment_dir}/state_dict_{savename}"
    )  # saves state_dict for loading with fastai
    learner.model.eval()
    learner.model.to("cpu")
    scripted_model = torch.jit.script(learner.model)
    torch.jit.save(
        scripted_model,
        f"{experiment_dir}/scripting_cpu_{savename}",
    )
    print(f"{experiment_dir}/scripting_cpu_{savename}")
    print(f"{experiment_dir}/state_dict_{savename}")
    return (
        f"{experiment_dir}/state_dict_{savename}",
        f"{experiment_dir}/scripting_cpu_{savename}",
    )


def load_tracing_model(savepath):
    tracing_model = torch.jit.load(savepath)
    return tracing_model


def test_tracing_model_one_batch(dls, tracing_model):
    x, _ = dls.one_batch()
    out_batch_logits = tracing_model(x)
    return out_batch_logits


def logits_to_classes(out_batch_logits):
    """returns the confidence scores of the max confident classes
    and an array of max confident class ids.
    """
    probs = torch.nn.functional.softmax(out_batch_logits.squeeze(), dim=0)
    conf, classes = torch.max(probs, 0)
    return conf, classes


def calculate_dice_coefficient(u, v):
    """
    Takes two pixel-confidence masks, and calculates how similar they are to each other
    Returns a value between 0 (no overlap) and 1 (identical)
    Utilizes an IoU style construction
    Can be used as NMS across classes for mutually-exclusive classifications
    """
    return 2 * torch.sum(torch.sqrt(torch.mul(u, v))) / (torch.sum(u + v))


def apply_conf_threshold_unet(conf, classes, conf_threshold):
    """Apply a confidence threshold to the output of logits_to_classes for a tile.
    Args:
        conf (np.ndarray): an array of shape [H, W] of max confidence scores for each pixel
        classes (np.ndarray): an array of shape [H, W] of class integers for the max confidence scores for each pixel
        conf_threshold (float): the threshold to use to determine whether a pixel is background or maximally confident category
    Returns:
        torch.Tensor: An array of shape [H,W] with the class ids that satisfy the confidence threshold. This can be vectorized.
    """
    high_conf_mask = torch.any(torch.where(conf > conf_threshold, 1, 0), axis=0)
    return torch.where(high_conf_mask, classes, 0)


def keep_by_global_pixel_nms(pred_dict, pixel_nms_thresh):
    """
    Apply non-maximum suppression (NMS) to a dictionary of predictions.

    This function iterates over a dictionary of predicted masks and calculates
    the Dice Coefficient to measure similarity between each pair of masks.
    If the coefficient exceeds a certain threshold, the mask is marked for removal.

    Args:
        pred_dict (dict): Dictionary with key "masks" containing a list of predicted masks.
        pixel_nms_thresh (float): The threshold above which two predictions are considered overlapping.

    Returns:
        list: List of indices of masks to keep.
    """
    masks = pred_dict["masks"]
    masks_to_remove = []

    for i, current_mask in enumerate(masks):  # Loop through all masks
        # Skip if the mask is already marked for removal
        if i in masks_to_remove:
            continue

        # Check similarity against all subsequent masks
        for j, comparison_mask in enumerate(masks[i + 1 :], start=i + 1):
            # Skip if the mask is already marked for removal
            if j in masks_to_remove:
                continue

            # Calculate Dice Coefficient; if the similarity is too high, mark mask for removal
            if (
                calculate_dice_coefficient(
                    current_mask.squeeze(), comparison_mask.squeeze()
                )
                > pixel_nms_thresh
            ):
                masks_to_remove.append(j)

    # Return a list of mask indices that are not marked for removal
    return [i for i in range(len(masks)) if i not in masks_to_remove]


def flatten_pixel_segmentation(pred_list, poly_score_thresh, shape):
    merged_pred_masks = []
    zeros = [torch.zeros(shape).long()]
    for pred_dict in pred_list:
        high_conf_classes = [
            torch.where(pred_dict["masks"][i] > poly_score_thresh, label, 0)
            .squeeze()
            .long()
            for i, label in enumerate(pred_dict["labels"])
        ]
        merged_pred_masks.extend(
            [torch.max(torch.dstack(zeros + high_conf_classes), axis=2)[0].numpy()]
        )  # TODO This max() compresses overlapping inferences naively
    return merged_pred_masks


def flatten_gt_segmentation(ds):
    merged_gt_masks = []
    zeros = [torch.zeros(ds[0].common.img.shape[:2]).long()]
    for record in ds:
        # Create mask for each class label in the record
        masks_gt = [
            record.detection.mask_array.data[j] * label_id
            for j, label_id in enumerate(record.detection.label_ids)
        ]
        # Merge all masks into one semantic mask with class labels (overlapping masks are compressed using max function)
        merged_gt_masks.extend([np.max(np.stack(zeros + masks_gt), axis=0)])
    return merged_gt_masks


def polygonize_pixel_segmentations(pred_dict, poly_score_thresh, bounds):
    high_conf_classes = [
        torch.where(mask > poly_score_thresh, pred_dict["labels"][i], 0)
        .squeeze()
        .long()
        for i, mask in enumerate(pred_dict["masks"])
    ]
    transform = (
        from_bounds(*bounds, *high_conf_classes[0].shape[:2])
        if high_conf_classes
        else None
    )
    pred_dict["polys"] = [
        vectorize_mask_instance(c, transform) for c in high_conf_classes
    ]
    keep_masks = [i for i, poly in enumerate(pred_dict["polys"]) if poly]
    return pred_dict, keep_masks


def create_memfile(high_conf_mask, transform):
    """Creates a raster in memory from a mask tensor."""

    memfile = MemoryFile()
    high_conf_mask = high_conf_mask.detach().numpy().astype("int16")

    with memfile.open(
        driver="GTiff",
        height=high_conf_mask.shape[0],
        width=high_conf_mask.shape[1],
        count=1,
        dtype=high_conf_mask.dtype,
        transform=transform,
        crs="EPSG:4326",
    ) as dataset:
        dataset.write(high_conf_mask, 1)

    return memfile


def extract_geometries(dataset):
    """Extracts the geometries from a raster dataset."""

    shps = shapes(
        dataset.read(1).astype("uint8"), connectivity=8, transform=dataset.transform
    )
    geoms = [shape(geom) for geom, value in shps if value != 0]
    return geoms


def vectorize_mask_instance(high_conf_mask: torch.Tensor, transform):
    """
    From a high confidence mask, generate a GeoJSON feature collection.

    Args:
        high_conf_mask (torch.Tensor): A tensor representing the high confidence mask.
        transform: A transformation to apply to the mask.

    Returns:
        geojson.Feature: A GeoJSON feature object.
    """

    memfile = create_memfile(high_conf_mask, transform)

    with memfile.open() as dataset:
        geoms = extract_geometries(dataset)
        multipoly = MultiPolygon(geoms)

    return (
        geojson.Feature(
            geometry=multipoly,
            properties=dict(),
        )
        if multipoly
        else None
    )


def keep_by_pixel_score(pred_dict, pixel_score_thresh):
    mask_maxes = (
        torch.stack([m.max() for m in pred_dict["masks"]])
        if len(pred_dict["masks"])
        else torch.tensor([])
    )
    return torch.where(mask_maxes > pixel_score_thresh)[0]


def keep_by_bbox_score(pred_dict, bbox_score_thresh):
    return torch.where(pred_dict["scores"] > bbox_score_thresh)[0]


def keep_boxes_by_idx(pred_dict, keep_idxs):
    if not len(keep_idxs):  # if keep_idxs is empty
        return {
            key: [] if isinstance(val, list) else torch.tensor([])
            for key, val in pred_dict.items()
        }
    else:
        keep_idxs = (
            torch.tensor(keep_idxs) if isinstance(keep_idxs, list) else keep_idxs
        )
        return {
            key: [val[i] for i in keep_idxs]
            if isinstance(val, list)
            else torch.index_select(val, 0, keep_idxs)
            for key, val in pred_dict.items()
        }


def reduce_preds(
    pred_list,
    bbox_score_thresh=None,
    pixel_score_thresh=None,
    pixel_nms_thresh=None,
    poly_score_thresh=None,
    bounds=[0, -1, 1, 0],
    **kwargs,
):
    """
    Apply various post-processing steps to the predictions from an object detection model.

    The post-processing includes:
    1. Removal of instances with bounding boxes below a certain confidence score.
    2. Removal of instances with pixel scores below a certain threshold.
    3. Application of non-maximum suppression at the pixel level.
    4. Generation of vectorized polygons from the remaining predictions.

    Arguments:
    - pred_list: A list of dictionaries containing model predictions. The dictionary should contain tensors with keys: "boxes", "labels", "scores", "masks".
    - bbox_score_thresh: A float indicating the confidence threshold for bounding boxes.
    - pixel_score_thresh: A float indicating the confidence threshold for pixels.
    - pixel_nms_thresh: A float indicating the threshold for pixel-based non-maximum suppression.
    - poly_score_thresh: A float indicating the confidence threshold for polygons.
    - bounds: A list representing the geographical bounds [west, south, east, north]. Default is [0, -1, 1, 0].
    - kwargs: Additional parameters.

    Returns:
    - A dictionary containing the post-processed predictions.
    """
    reduced_preds = []
    for pred_dict in tqdm(pred_list):
        # remove instances with low scoring boxes
        if bbox_score_thresh is not None:
            keep = keep_by_bbox_score(pred_dict, bbox_score_thresh)
            pred_dict = keep_boxes_by_idx(pred_dict, keep)

        # remove instances with low scoring pixels
        if pixel_score_thresh is not None:
            keep = keep_by_pixel_score(pred_dict, pixel_score_thresh)
            pred_dict = keep_boxes_by_idx(pred_dict, keep)

        # non-maximum suppression, done globally, across classes, using pixels rather than bboxes
        if pixel_nms_thresh is not None:
            keep = keep_by_global_pixel_nms(pred_dict, pixel_nms_thresh)
            pred_dict = keep_boxes_by_idx(pred_dict, keep)

        # generate vectorized polygons from predictions
        # adds "polys" to pred_dict
        if poly_score_thresh is not None:
            pred_dict, keep = polygonize_pixel_segmentations(
                pred_dict, poly_score_thresh, bounds
            )
            pred_dict = keep_boxes_by_idx(pred_dict, keep)

        reduced_preds.extend([pred_dict])
    return reduced_preds


def raw_predict(scripted_model, eval_ds):
    raw_preds = []
    for record in tqdm(eval_ds):
        pred_dict = scripted_model([torch.Tensor(np.moveaxis(record.img, 2, 0)) / 255])[
            1
        ][0]
        raw_preds.extend([{k: v.detach() for k, v in pred_dict.items()}])
    return raw_preds
