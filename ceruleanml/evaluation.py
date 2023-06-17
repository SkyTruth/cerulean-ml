import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from icevision.core.mask import RLE
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

from ceruleanml.inference import (
    apply_conf_threshold_unet,
    calculate_dice_coefficient,
    logits_to_classes,
)

mpl.rcParams["axes.grid"] = False
mpl.rcParams["figure.figsize"] = (12, 12)


def cm_f1(
    arrays_gt,
    arrays_pred,
    save_dir,
    normalize=None,
    class_names=[],
    title="Confusion Matrix",
):
    """Takes paired arrays for ground truth and predicition masks, as well
        as the number of target classes and a directory to save the
        normalized confusion matrix plot to.
    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Args:
        arrays_gt (np.ndarray): The ground truth mask array from a label tile.
            Can have 1 channel shaped like (H, W, Channels).
        arrays_pred (np.ndarray): The prediction mask array from a validation tile
            having undergone inference.
            Can have 1 channel shaped like (H, W, Channels).
        save_dir (string): The output directory to write the normalized confusion matrix plot to.
    Returns:
        F1 score (float): Evaluation metric. Harmonic mean of precision and recall.
        Normalized confusion matrix (table): The confusion matrix table.
    """
    # flatten our mask arrays and use scikit-learn to create a confusion matrix
    flat_preds = np.concatenate(arrays_pred).flatten()
    flat_truth = np.concatenate(arrays_gt).flatten()
    OUTPUT_CHANNELS = len(class_names)
    cm = confusion_matrix(
        flat_truth, flat_preds, labels=list(range(OUTPUT_CHANNELS)), normalize=normalize
    )

    # cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    if not class_names:
        xticklabels = list(range(OUTPUT_CHANNELS))
        yticklabels = list(range(OUTPUT_CHANNELS))
    else:
        assert len(class_names) > 1
        xticklabels = class_names
        yticklabels = class_names
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f"  # 'd' # if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    ax.set_ylim(len(class_names) - 0.5, -0.5)
    if normalize == "true":
        cm_name = os.path.join(f"{save_dir}", "cm_normed_true.png")
    elif normalize == "pred":
        cm_name = os.path.join(f"{save_dir}", "cm_normed_pred.png")
    elif normalize is None:
        cm_name = os.path.join(f"{save_dir}", "cm_count.png")
    else:
        raise ValueError(
            "normalize is not pred, true or None, check cm docs for sklearn."
        )

    plt.savefig(cm_name)
    print(f"Confusion matrix saved at {cm_name}")
    # compute f1 score
    f1 = f1_score(flat_truth, flat_preds, average="macro")
    print("f1_score", f1)

    return cm, f1


def get_cm_for_torchscript_model_unet(
    dls,
    model,
    save_path,
    semantic_mask_conf_thresh,
    normalize=None,
    class_names=[],
    title="Confusion Matrix",
):
    """
    the torchscript model when it is loaded operates on batches, not individual images
    this doesn't support eval on negative samples if they are in the dls,
    since val masks don't exist with neg samples. need to be constructed with np.zeros

    returns cm and f1 score
    """
    val_arrs = []
    class_preds = []
    for batch_tuple in tqdm(dls.valid):
        semantic_masks_batch = batch_tuple[1].cpu().detach().numpy()
        class_pred_batch = model(batch_tuple[0].cpu())
        probs, classes = logits_to_classes(class_pred_batch)
        t = apply_conf_threshold_unet(probs, classes, semantic_mask_conf_thresh)
        class_pred_batch = t.cpu().detach().numpy()
        val_arrs.extend(semantic_masks_batch)
        class_preds.append(class_pred_batch)
    return cm_f1(
        val_arrs,
        class_preds,
        save_path,
        normalize,
        class_names,
        title=title,
    )


def gt_bbox_dice(target, prediction):
    # Convert bounding boxes to tensors
    stacked_preds = prediction["boxes"] or torch.empty(0, 4)

    stacked_targets = [bbox.to_tensor() for bbox in target.detection.bboxes]
    stacked_targets = (
        torch.stack(stacked_targets) if stacked_targets else torch.empty(0, 4)
    )

    # Calculate Intersection-over-Union for each pair of prediction and target
    return torchvision.ops.box_iou(stacked_preds, stacked_targets)


def gt_pixel_dice(target, prediction):
    # Initialize a zero tensor for results
    res = torch.zeros(len(prediction["masks"]), len(target.detection.masks))

    # Loop through each mask in the prediction and target
    for i, pred_mask in enumerate(prediction["masks"]):
        for j, target_mask in enumerate(target.detection.masks):
            # If the mask is in RLE format, convert it to a tensor mask
            target_mask = (
                rle_to_tensor_mask(target_mask, target.common.img_size)
                if type(target_mask) == RLE
                else target_mask
            )

            # Calculate the soft dice coefficient for the pair of masks
            res[i, j] = calculate_dice_coefficient(
                torch.squeeze(pred_mask), torch.squeeze(target_mask)
            )
    return res


def rle_to_tensor_mask(rle_mask, resize=None):
    mask_tile_size = int(np.sqrt(sum(rle_mask.counts)))
    if mask_tile_size**2 != sum(rle_mask.counts):
        # target tile is not square, and we're not sure of the original dimensions...
        raise NotImplementedError
    mask_tensor = rle_mask.to_mask(mask_tile_size, mask_tile_size).to_tensor().squeeze()
    if resize and resize != mask_tensor.size():
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor[None, None, ...], size=resize, mode="nearest"
        )[0, 0, :]
    return mask_tensor
