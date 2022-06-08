"""Adapted from https://towardsdatascience.com/how-to-create-a-datablock-for-multispectral-satellite-image-segmentation-with-the-fastai-v2-bc5e82f4eb5
This module adds Background value masking (assumed 0) for plotting masks on top of multispectral images.
It's also adapted to work with the record collection object returned by icevision's COCO parser instead of files in folders.
The multispectral image is assumed to have channels last in the dim order."""

import numpy as np
from fastai.vision.core import TensorImage, TensorMask
from numpy import ndarray
import skimage.io as skio
import matplotlib as mpl
import matplotlib.pyplot as plt
from ceruleanml.coco_load_fastai import get_image_path, record_to_mask
import torch


def open_n_channel_img(fn, chnls=None, cls=torch.Tensor):
    im = torch.from_numpy(
        skio.imread(str(fn))
    )  # datablock expects at least 1 channel not an absent channel dimension so we use newaxis
    if chnls is not None:
        im = im[..., chnls]
    if len(im.shape) == 2 or im.shape[-1] == 1:
        im = im.squeeze()
        return cls(im.type(torch.float32))[None, :, :]
    else:
        return cls(im.type(torch.float32))


def open_mask_from_record(record_id, record_collection, cls=torch.Tensor):
    arr = record_to_mask(record_collection, record_id)
    im = torch.from_numpy(arr).type(torch.float32)
    return cls(im)


class MSTensorImage(TensorImage):
    @classmethod
    def create(cls, data: (int, ndarray), record_collection, chnls=None):
        fn = get_image_path(record_collection, data)
        im = open_n_channel_img(fn=fn, chnls=chnls, cls=torch.Tensor)
        return cls(im)

    def show(self, chnls=[3, 2, 1], ctx=None, **kwargs):
        visu_img = self
        if visu_img.ndim > 2:
            visu_img = self[chnls, ...]
        else:
            visu_img = self
        visu_img = visu_img.squeeze()
        plt.imshow(visu_img) if ctx is None else ctx.imshow(visu_img)
        return ctx

    # def __repr__(self):

    #     return f"MSTensorImage: {self.shape}"


class MSTensorMask(TensorMask):
    @classmethod
    def create(cls, data: (int, ndarray), record_collection):
        im = open_mask_from_record(data, record_collection, cls=torch.Tensor)
        return cls(im)

    def show(self, ctx=None, show_label=True, **kwargs):
        visu_img = self
        if show_label:
            visu_img = visu_img.squeeze()
            visu_img = np.ma.masked_where(visu_img == 0, visu_img)
            cmap = "Set2"
            plt.imshow(visu_img, cmap=cmap, interpolation="none") if ctx is None else ctx.imshow(
                visu_img, cmap=cmap, vmin=1, vmax=6, interpolation="none"
            )
        else:
            pass
        return ctx

    # def __repr__(self):

    #     return f"MSTensorImage: {self.shape}"
