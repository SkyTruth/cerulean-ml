from icevision import models, parsers, show_records, tfms, Dataset, Metric, COCOMetric, COCOMetricType
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
 
from icevision.imports import *
from icevision.utils import *
from icevision.data import *
from icevision.metrics.metric import *

import numpy as np

import json
import os

__all__ = ["IoUMetric", "IoUMetricType"]


class IoUMetricType(Enum):
    """Available options for `COCOMetric`."""

    bbox = "bbox"
    mask = "segm"
    keypoint = "keypoints"


class IoUMetric(Metric):
    """Wrapper around [cocoapi evaluator](https://github.com/cocodataset/cocoapi)
    Calculates average precision.
    # Arguments
        metric_type: Dependent on the task you're solving.
        print_summary: If `True`, prints a table with statistics.
        show_pbar: If `True` shows pbar when preparing the data for evaluation.
    """

    def __init__(
        self,
        metric_type: IoUMetricType = IoUMetricType.bbox,
        iou_thresholds: Optional[Sequence[float]] = np.linspace(.05, 0.95, int(np.round((0.95 - .05) / .05)) + 1, endpoint=True), #None,
        print_summary: bool = False, 
        show_pbar: bool = True,
    ):
        self.metric_type = metric_type
        self.iou_thresholds = iou_thresholds
        self.print_summary = print_summary
        self.show_pbar = show_pbar
        self._records, self._preds = [], []

    def _reset(self):
        self._records.clear()
        self._preds.clear()

    def accumulate(self, preds):
        for pred in preds:
            self._records.append(pred.ground_truth)
            self._preds.append(pred.pred)

    def finalize(self) -> Dict[str, float]:
        with CaptureStdout():
            coco_eval = create_coco_eval(
                records=self._records,
                preds=self._preds,
                metric_type=self.metric_type.value,
                iou_thresholds=self.iou_thresholds,
                show_pbar=self.show_pbar,
            )
            coco_eval.evaluate()
            coco_eval.accumulate()

        with CaptureStdout(propagate_stdout=self.print_summary):
            coco_eval.summarize()

        stats = coco_eval.stats
        ious = coco_eval.ious

        ious_l = []
        for iou in ious.values():
            if isinstance(iou, np.ndarray):
                iou = iou.tolist()
            else:
                iou = iou
            ious_l.append(iou)
        
        flat_ious_l = [item for sublist in ious_l for item in sublist]
        if len(flat_ious_l) == 0:
            flat_ious_l.append([0])
        flat_ious_l = [item for items in flat_ious_l for item in items]
        ious_avg = np.array(flat_ious_l).mean()
        ious_min = np.array(flat_ious_l).min()
        ious_max = np.array(flat_ious_l).max()
        
        logs = {
            #"Min IoU area=all": ious_min,
            #"Max IoU area=all": ious_max,
            "Avg. IoU area=all": ious_avg, 
        }
        self._reset()
        
        
        return logs
