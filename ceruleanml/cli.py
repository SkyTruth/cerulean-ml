"""Console script for ceruleanml."""
import os
import sys
from datetime import date
from pathlib import Path
from typing import List

import click

import ceruleanml.data as data


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
        {"supercategory": "slick", "id": i, "name": cname}
        for i, cname in enumerate(class_list)
    ]  # order matters, check that this matches the ids used when annotating if you get a data loading error
    return {
        "info": info,
        "licenses": licenses,
        "images": [],
        "annotations": [],
        "categories": categories,
    }


@click.command()
def main(args=None):
    """Console script for ceruleanml."""
    click.echo("Replace this message by putting your code into " "ceruleanml.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


@click.command()
def make_coco_dataset_with_tiles(
    class_folder_path: str,
    aux_data_path: str,
    coco_outdir: str,
    name: str = "Tiled Cerulean Dataset V2",
):
    """_summary_

    Args:
        class_folder_path (str): _description_
        aux_data_path (str): _description_
        coco_outdir (str): _description_
    """
    os.makedirs(coco_outdir, exist_ok=True)
    os.makedirs(os.path.join(coco_outdir, "tiled_images"), exist_ok=True)
    class_foldes_path = Path(class_folder_path)
    class_folders = list(class_foldes_path.glob("*/"))
    coco_output = make_coco_metadata(name=name)
    coco_tiler = data.COCOtiler(os.path.join(coco_outdir, "tiled_images"), coco_output)

    aux_datasets = [os.path.join(aux_data_path, "infra_locations.json"), "ship_density"]
    for class_folder in class_folders:
        for scene_folder in list(class_folder.glob("*GRDH*")):
            assert "S1" in str(scene_folder)
            print(scene_folder)
            scene_id = os.path.basename(scene_folder)
            layer_pths = [str(i) for i in list(scene_folder.glob("*png"))]
            print(layer_pths)
            coco_tiler.save_background_img_tiles(
                scene_id,
                layer_pths,
                aux_datasets=aux_datasets,
                aux_resample_ratio=8,
            )
            coco_tiler.create_coco_from_photopea_layers(
                scene_id, layer_pths, coco_output
            )
    coco_tiler.save_coco_output(
        os.path.join(coco_outdir, f"./instances_{name.replace('', '')}.json")
    )
    print(f"Images and COCO JSON have been saved in {coco_outdir}.")


@click.command()
def make_coco_dataset_no_tiles(
    class_folder_path: str,
    aux_data_path: str,
    coco_outdir: str,
    name="Untiled Cerulean Dataset V2 No Context Files",
):
    """_summary_

    Args:
        class_folder_path (str): _description_
        aux_data_path (str): _description_
        coco_outdir (str): _description_
    """
    os.makedirs(coco_outdir, exist_ok=True)
    os.makedirs(os.path.join(coco_outdir, "untiled_images"), exist_ok=True)
    class_foldes_path = Path(class_folder_path)
    class_folders = list(class_foldes_path.glob("*/"))
    coco_output = make_coco_metadata(name=name)
    coco_tiler = data.COCOtiler(
        os.path.join(coco_outdir, "untiled_images"), coco_output
    )
    coco_tiler.copy_background_images([str(i) for i in class_folders])
    for class_folder in class_folders:
        for scene_folder in list(class_folder.glob("*GRDH*")):
            assert "S1" in str(scene_folder)
            scene_id = os.path.basename(scene_folder)
            layer_pths = [str(i) for i in list(scene_folder.glob("*png"))]
            coco_tiler.create_coco_from_photopea_layers_no_tile(
                scene_id, layer_pths, coco_output
            )
    coco_tiler.save_coco_output(
        os.path.join(coco_outdir, f"./instances_{name.replace('', '')}.json")
    )
    print(f"Images and COCO JSON have been saved in {coco_outdir}.")


@click.command()
def make_coco_dataset_no_context(
    class_folder_path: str,
    aux_data_path: str,
    coco_outdir: str,
    name="Tiled Cerulean Dataset V2 No Context Files",
):
    """_summary_

    Args:
        class_folder_path (str): _description_
        aux_data_path (str): _description_
        coco_outdir (str): _description_
    """
    os.makedirs(coco_outdir, exist_ok=True)
    os.makedirs(os.path.join(coco_outdir, "tiled_images_no_context"), exist_ok=True)
    class_foldes_path = Path(class_folder_path)
    class_folders = list(class_foldes_path.glob("*/"))
    coco_output = make_coco_metadata(name=name)
    coco_tiler = data.COCOtiler(
        os.path.join(coco_outdir, "tiled_images_no_context"), coco_output
    )
    for class_folder in class_folders:
        for scene_folder in list(class_folder.glob("*GRDH*")):
            assert "S1" in str(scene_folder)
            scene_id = os.path.basename(scene_folder)
            layer_pths = [str(i) for i in list(scene_folder.glob("*png"))]
            coco_tiler.save_background_img_tiles(
                scene_id,
                layer_pths,
                aux_datasets=[],
                aux_resample_ratio=8,
            )
            coco_tiler.create_coco_from_photopea_layers(
                scene_id, layer_pths, coco_output
            )
    coco_tiler.save_coco_output(
        os.path.join(coco_outdir, f"./instances_{name.replace('', '')}.json")
    )
    print(f"Images and COCO JSON have been saved in {coco_outdir}.")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
