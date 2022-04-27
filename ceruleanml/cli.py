"""Console script for ceruleanml."""
import os
from datetime import date
from pathlib import Path
from typing import List

import click
from tqdm import tqdm

import ceruleanml.data as data
import dask
import time
from dask.distributed import Client, progress


@click.group()
def main():
    """CeruleanML CLI scripts for data processing."""


def make_coco_metadata(
    name="Cerulean Dataset V2",
    description: str = "Cerulean Dataset V2",
    version: str = "1.0",
    class_list: List[str] = [
        "infra_slick",
        "natural_seep",
        "coincident_vessel",
        "recent_vessel",
        "old_vessel",
        "ambiguous",
    ],
):
    """Creates COCO Metadata

    Args:
        dname (str, optional): The name of the dataset. Defaults to "Cerulean Dataset V2".
        description (str, optional): A string description fo the coco dataset. Defaults to "Cerulean Dataset V2".
        version (str, optional): Defaults to "1.0".
        class_list (List, optional): The lsit of classes. Ordering of list must match integer ids used during annotation. Defaults to [ "infra_slick", "natural_seep", "coincident_vessel", "recent_vessel", "old_vessel", "ambiguous", ].

    Returns:
        dict : A COCO metadata dictionary.
    """
    info = {
        "description": description,
        "url": "none",
        "version": version,
        "year": 2022,
        "contributor": "Skytruth",
        "date_created": str(date.today()),
    }
    licenses = [{"url": "none", "id": 1, "name": name}]
    categories = [
        {"supercategory": "slick", "id": i + 1, "name": cname} for i, cname in enumerate(class_list)
    ]  # order matters, check that this matches the ids used when annotating if you get a data loading error
    return {
        "info": info,
        "licenses": licenses,
        "images": [],
        "annotations": [],
        "categories": categories,
    }


@main.command()
@click.argument("class_folder_path", nargs=1)
@click.argument("aux_data_path", nargs=1)
@click.argument("coco_outdir", nargs=1)
def make_coco_dataset_with_tiles(
    class_folder_path: str,
    aux_data_path: str,
    coco_outdir: str,
    name: str = "Tiled Cerulean Dataset V2",
):
    """Create the dataset with tiles and context files (ship density and infra distance).

    Args:
        class_folder_path (str): the path to the folder containing class
            folders ("Infrastructure", "Coincident", etc.)
        aux_data_path (str): the path to the folder containing the aux
            files (currently only infra_locations.json)
        coco_outdir (str): the path to save the coco json and the folder
            of tiled images.
    """
    client = Client()  # this needs to be commented out to use single threaded for profiling
    start = time.time()
    os.makedirs(coco_outdir, exist_ok=True)
    os.makedirs(os.path.join(coco_outdir, "tiled_images"), exist_ok=True)
    class_foldes_path = Path(class_folder_path)
    class_folders = list(class_foldes_path.glob("*/"))
    coco_tiler = data.COCOtiler(os.path.join(coco_outdir, "tiled_images"))

    aux_datasets = [os.path.join(aux_data_path, "infra_locations.json"), "ship_density"]
    scene_index = 0
    coco_outputs = []
    for class_folder in class_folders:
        for scene_folder in list(class_folder.glob("*GRDH*"))[0:1]:
            assert "S1" in str(scene_folder)
            scene_id = os.path.basename(scene_folder)
            layer_pths = [str(i) for i in list(scene_folder.glob("*png"))]
            scene_data_tuple = dask.delayed(coco_tiler.save_background_img_tiles)(
                scene_id,
                layer_pths,
                aux_datasets=aux_datasets,
                aux_resample_ratio=8,
            )
            coco_output = dask.delayed(coco_tiler.create_coco_from_photopea_layers)(
                scene_index, scene_data_tuple, layer_pths
            )
            coco_outputs.append(coco_output)
            scene_index += 1
    final_coco_output = make_coco_metadata(name=name)
    # when we create a distributed client, dask.compute uses that isntead of thread scheduler by default
    coco_outputs = dask.persist(*coco_outputs)  # start computation in the background
    progress(coco_outputs)  # watch progress
    coco_outputs = dask.compute(*coco_outputs)
    # coco_outputs = dask.compute(
    #     *coco_outputs, scheduler="single-threaded"
    # )  # convert to final result when done
    for co in coco_outputs:
        final_coco_output["images"].extend(co["images"])
        final_coco_output["annotations"].extend(co["annotations"])
    coco_tiler.save_coco_output(
        final_coco_output, os.path.join(coco_outdir, f"./instances_{name.replace('', '')}.json")
    )
    num_images = len(final_coco_output["images"])
    print(f"Number of seconds for {num_images} images: {time.time() - start}")
    print(f"Images and COCO JSON have been saved in {coco_outdir}.")


@main.command()
@click.argument("class_folder_path", nargs=1)
@click.argument("coco_outdir", nargs=1)
def make_coco_dataset_no_tiles(
    class_folder_path: str,
    coco_outdir: str,
    name="Untiled Cerulean Dataset V2 No Context Files",
):
    """Create the dataset without tiling and without context files.

    Args:
        class_folder_path (str): the path to the folder containing class
            folders ("Infrastructure", "Coincident", etc.)
        coco_outdir (str): the path to save the coco json and the folder
            of tiled images.
    """
    os.makedirs(coco_outdir, exist_ok=True)
    os.makedirs(os.path.join(coco_outdir, "untiled_images"), exist_ok=True)
    class_foldes_path = Path(class_folder_path)
    class_folders = list(class_foldes_path.glob("*/"))
    coco_output = make_coco_metadata(name=name)
    coco_tiler = data.COCOtiler(os.path.join(coco_outdir, "untiled_images"), coco_output)
    coco_tiler.copy_background_images([str(i) for i in class_folders])
    for class_folder in class_folders:
        for scene_folder in list(class_folder.glob("*GRDH*")):
            assert "S1" in str(scene_folder)
            scene_id = os.path.basename(scene_folder)
            layer_pths = [str(i) for i in list(scene_folder.glob("*png"))]
            coco_tiler.create_coco_from_photopea_layers_no_tile(scene_id, layer_pths)
    coco_tiler.save_coco_output(
        os.path.join(coco_outdir, f"./instances_{name.replace('', '')}.json")
    )
    print(f"Images and COCO JSON have been saved in {coco_outdir}.")


@main.command()
@click.argument("class_folder_path", nargs=1)
@click.argument("coco_outdir", nargs=1)
def make_coco_dataset_no_context(
    class_folder_path: str,
    coco_outdir: str,
    name="Tiled Cerulean Dataset V2 No Context Files",
):
    """Create the dataset with tiles but without context files (ship density and infra distance).

    Args:
        class_folder_path (str): the path to the folder containing class
            folders ("Infrastructure", "Coincident", etc.)
        coco_outdir (str): the path to save the coco json and the folder
            of tiled images.
    """
    os.makedirs(coco_outdir, exist_ok=True)
    os.makedirs(os.path.join(coco_outdir, "tiled_images_no_context"), exist_ok=True)
    class_foldes_path = Path(class_folder_path)
    class_folders = list(class_foldes_path.glob("*/"))
    coco_output = make_coco_metadata(name=name)
    coco_tiler = data.COCOtiler(os.path.join(coco_outdir, "tiled_images_no_context"), coco_output)
    for class_folder in tqdm(class_folders):
        for scene_folder in tqdm(list(class_folder.glob("*GRDH*"))):
            assert "S1" in str(scene_folder)
            scene_id = os.path.basename(scene_folder)
            layer_pths = [str(i) for i in list(scene_folder.glob("*png"))]
            coco_tiler.save_background_img_tiles(
                scene_id,
                layer_pths,
                aux_datasets=[],
                aux_resample_ratio=8,
            )
            coco_tiler.create_coco_from_photopea_layers(scene_id, layer_pths)
    coco_tiler.save_coco_output(
        os.path.join(coco_outdir, f"./instances_{name.replace('', '')}.json")
    )
    print(f"Images and COCO JSON have been saved in {coco_outdir}.")


if __name__ == "__main__":
    main()
