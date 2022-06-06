import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

mpl.rcParams["axes.grid"] = False
mpl.rcParams["figure.figsize"] = (12, 12)


def cm_f1(arrays_gt, arrays_pred, num_classes, save_dir, normalize=None):
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
        num_classes (integer): The number of target classes.
        save_dir (string): The output directory to write the normalized confusion matrix plot to.
    Returns:
        F1 score (float): Evaluation metric. Harmonic mean of precision and recall.
        Normalized confusion matrix (table): The confusion matrix table.
    """
    # flatten our mask arrays and use scikit-learn to create a confusion matrix
    flat_preds = np.concatenate(arrays_gt).flatten()
    flat_truth = np.concatenate(arrays_pred).flatten()
    OUTPUT_CHANNELS = num_classes
    cm = confusion_matrix(
        flat_truth, flat_preds, labels=list(range(OUTPUT_CHANNELS)), normalize=normalize
    )

    classes = list(range(0, num_classes))

    # cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=list(range(OUTPUT_CHANNELS)),
        yticklabels=list(range(OUTPUT_CHANNELS)),
        title="Confusion Matrix",
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
    ax.set_ylim(len(classes) - 0.5, -0.5)

    cm_name = (
        f"{save_dir}/cm_normed.png"
        if normalize is not None
        else f"{save_dir}/cm_count.png"
    )
    plt.savefig(cm_name)

    # compute f1 score
    f1 = f1_score(flat_truth, flat_preds, average="macro")

    return cm, f1
