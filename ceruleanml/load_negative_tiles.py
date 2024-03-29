import mimetypes
import os
from pathlib import Path
from typing import Any, Hashable

from fastcore.basics import setify
from fastcore.foundation import L
from icevision.core import record_defaults
from icevision.core.class_map import ClassMap
from icevision.data import SingleSplitSplitter
from icevision.parsers.parser import Parser
from icevision.utils.get_files import get_image_files
from icevision.utils.imageio import get_img_size

from ceruleanml.coco_load_fastai import get_image_path

# https://airctic.com/dev/negative_samples/
negative_template_record = record_defaults.InstanceSegmentationRecord()
# Parser.generate_template(negative_template_record)


class NegativeImageParser(Parser):
    def __init__(self, template_record, data_dir, images_positive, count, class_names):
        super().__init__(template_record=template_record)
        self.class_map = ClassMap(class_names)
        self.image_filepaths = get_negative_image_files(
            data_dir, images_positive, count
        )  # get_image_files(data_dir)

    def __iter__(self) -> Any:
        yield from self.image_filepaths

    def __len__(self) -> int:
        return len(self.image_filepaths)

    def record_id(self, o) -> Hashable:
        return o.stem

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_img_size(get_img_size(o))
            record.set_filepath(o)
            record.detection.set_class_map(self.class_map)


__all__ = ["get_files", "get_image_files"]


# All copied from fastai
def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(
    path,
    extensions=None,
    recurse=True,
    folders=None,
    followlinks=True,
    sort: bool = True,
):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified. From fastai"
    path = Path(path)
    folders = L(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path, followlinks=followlinks)
        ):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            if len(folders) != 0 and i == 0 and "." not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)

    return L(sorted(res)) if sort else L(res)


image_extensions = set(
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
)
# TODO docstring


def get_negative_image_files(path, images_positive, count, recurse=True, folders=None):
    "Get image files in `path` recursively, only in `folders`, if specified. From fastai"
    images_initial = get_files(
        path, extensions=image_extensions, recurse=recurse, folders=folders
    )
    images_negative = list(set(images_initial) - set(images_positive))
    return images_negative[0:count]


# TODO docstring
def parse_negative_tiles(data_dir, record_ids, positive_records, count, class_names):
    images_positive = []

    def get_image_by_record_id(record_id):
        return get_image_path(positive_records, record_id)

    # def get_mask_by_record_id(record_id):
    #     return record_to_mask(positive_records, record_id)

    for i in record_ids:
        im = get_image_by_record_id(i)
        images_positive.append(im)
    # TODO callout negative tiles not sampled they are indexed with count
    negative_parser = NegativeImageParser(
        negative_template_record,
        data_dir=data_dir,
        images_positive=images_positive,
        count=count,
        class_names=class_names,
    )
    negative_records = negative_parser.parse(data_splitter=SingleSplitSplitter())
    return negative_records[0]  # single split comes nested so we get the actual record
