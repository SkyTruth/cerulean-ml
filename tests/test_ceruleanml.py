#!/usr/bin/env python

"""Tests for `ceruleanml` package."""

import os
import pickle
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import skimage.io as skio
from click.testing import CliRunner

from ceruleanml import cli, data


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "ceruleanml.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


def mock_scene_info():
    return {
        "mocked": True,
        "bounds": [55.698181, 24.565813, 58.540211, 26.494711],
        "band_metadata": [["vv", {}], ["vh", {}]],
        "band_descriptions": [["vv", ""], ["vh", ""]],
        "dtype": "uint16",
        "nodata_type": "Nodata",
        "colorinterp": ["gray", "gray"],
    }


def test_handle_aux_datasets():
    coco_tiler = data.COCOtiler("", {})

    ar = coco_tiler.handle_aux_datasets(
        [
            "tests/fixtures/oil_areas_inverted_clip.geojson",
            "tests/fixtures/infra_locations_clip.geojson",
        ],
        scene_id="S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5",
        bounds=[
            55.69982872351191,
            24.566447533809654,
            58.53597315567021,
            26.496758065384803,
        ],
        image_shape=(4181, 6458),
    )
    assert ar.shape == (4181, 6458, 2)


def test_get_dist_array_from_vector():

    arr = data.get_dist_array_from_vector(
        bounds=(55.698181, 24.565813, 58.540211, 26.494711),
        img_shape=(4181, 6458),
        vector_ds="tests/fixtures/oil_areas_inverted_clip.geojson",
        aux_resample_ratio=10,
    )
    assert arr.shape == (4181, 6458)
    assert arr.dtype == np.dtype(np.uint8)
    assert np.max(arr) == 255
    assert np.min(arr) == 0


def test_dist_array_from_layers_points():

    arr = data.get_dist_array_from_vector(
        bounds=(55.698181, 24.565813, 58.540211, 26.494711),
        img_shape=(4181, 6458),
        vector_ds="tests/fixtures/infra_locations_clip.geojson",
        aux_resample_ratio=10,
    )
    assert arr.shape == (4181, 6458)
    assert arr.dtype == np.dtype(np.uint8)
    assert np.max(arr) == 255
    assert np.min(arr) == 0


def test_get_ship_density(httpx_mock):
    with open(
        "tests/fixtures/MLXF_ais__sq_07a7fea65ceb3429c1ac249f4187f414_9c69e5b4361b6bc412a41f85cdec01ee.zip",
        "rb",
    ) as src:
        httpx_mock.add_response(content=src.read())
    arr = data.get_ship_density(
        bounds=(55.698181, 24.565813, 58.540211, 26.494711), img_shape=(4181, 6458)
    )
    assert arr.shape == (4181, 6458)
    assert arr.dtype == np.dtype(np.uint8)
    assert np.max(arr) == 255
    assert np.min(arr) == 0


@pytest.mark.skip(reason="This requires connecting to S3 and requester pays bucket")
def test_fetch_sentinel1_reprojection_parameters():
    scene_id = "S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5"
    (
        wgs84_bounds,
        img_shape,
        gcps_transform,
        crs,
        rescale,
    ) = data.fetch_sentinel1_reprojection_parameters(scene_id)

    print(wgs84_bounds, img_shape, gcps_transform.to_gdal(), crs)
    # Results are almost equal to 2 decimals, better to use wgs84 bounds
    np.testing.assert_almost_equal(wgs84_bounds, data.get_sentinel1_bounds(scene_id), 2)
    assert img_shape == (20191, 29666)
    assert False


@patch("ceruleanml.data.fetch_sentinel1_reprojection_parameters")
def test_save_background_img_tiles(mock_fetch_sentinel_1_reprojection_parameters):
    scene_id = "S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5"
    with open("tests/fixtures/gcps_s1.xyz", "rb") as src:
        gcps = pickle.load(src)

    mock_fetch_sentinel_1_reprojection_parameters.return_value = (
        [55.69982872351191, 24.566447533809654, 58.53597315567021, 26.496758065384803],
        (int(20191 / 8), int(29666 / 8)),
        gcps,
        rasterio.crs.CRS.from_epsg(4326),
    )

    info = {
        "description": "Cerulean Dataset V2",
        "url": "none",
        "version": "1.0",
        "year": 2021,
        "contributor": "Skytruth",
        "date_created": "2022/2/23",
    }

    licenses = [{"url": "none", "id": 1, "name": "CeruleanDataset V2"}]
    categories = [
        {"supercategory": "slick", "id": 1, "name": "infra_slick"},
        {"supercategory": "slick", "id": 2, "name": "natural_seep"},
        {"supercategory": "slick", "id": 3, "name": "coincident_vessel"},
        {"supercategory": "slick", "id": 4, "name": "recent_vessel"},
        {"supercategory": "slick", "id": 5, "name": "old_vessel"},
        {"supercategory": "slick", "id": 6, "name": "ambiguous"},
    ]

    coco_output = {
        "info": info,
        "licenses": licenses,
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        coco_tiler = data.COCOtiler(tmp_dir, coco_output)

        class_file = f"tests/fixtures/{scene_id}/cv2_transfer_outputs_skytruth_annotation_first_phase_old_vessel_{scene_id}_ambiguous_1.png"
        # template = skio.imread(class_file)
        background_file = class_file.replace("ambiguous_1", "Background")
        # skio.imsave(background_file, template[:, :, 0])
        layer_paths = [background_file, class_file]

        # Pass same vector dataset twice to make RGB image
        coco_tiler.save_background_img_tiles(
            scene_id,
            layer_paths,
            aux_datasets=[
                "tests/fixtures/oil_areas_inverted_clip.geojson",
                "tests/fixtures/oil_areas_inverted_clip.geojson",
            ],
            aux_resample_ratio=100,
        )

        # os.remove(background_file)
        assert len(os.listdir(tmp_dir)) == 40


def test_create_coco_from_photopea_layers():
    scene_id = "S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5"
    with open("tests/fixtures/gcps_s1.xyz", "rb") as src:
        gcps = pickle.load(src)

    info = {
        "description": "Cerulean Dataset V2",
        "url": "none",
        "version": "1.0",
        "year": 2021,
        "contributor": "Skytruth",
        "date_created": "2022/2/23",
    }

    licenses = [{"url": "none", "id": 1, "name": "CeruleanDataset V2"}]
    categories = [
        {"supercategory": "slick", "id": 1, "name": "infra_slick"},
        {"supercategory": "slick", "id": 2, "name": "natural_seep"},
        {"supercategory": "slick", "id": 3, "name": "coincident_vessel"},
        {"supercategory": "slick", "id": 4, "name": "recent_vessel"},
        {"supercategory": "slick", "id": 5, "name": "old_vessel"},
        {"supercategory": "slick", "id": 6, "name": "ambiguous"},
    ]

    coco_output = {
        "info": info,
        "licenses": licenses,
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        coco_tiler = data.COCOtiler(tmp_dir, coco_output)
        coco_tiler.s1_scene_id = scene_id
        coco_tiler.s1_bounds = [
            55.69982872351191,
            24.566447533809654,
            58.53597315567021,
            26.496758065384803,
        ]
        coco_tiler.s1_image_shape = (int(20191 / 8), int(29666 / 8))
        coco_tiler.s1_gcps = gcps
        coco_tiler.s1_crs = rasterio.crs.CRS.from_epsg(4326)

        class_file = f"tests/fixtures/{scene_id}/cv2_transfer_outputs_skytruth_annotation_first_phase_old_vessel_{scene_id}_ambiguous_1.png"
        template = skio.imread(class_file)
        background_file = class_file.replace("ambiguous_1", "Background")
        skio.imsave(background_file, template[:, :, 0])
        layer_path = [background_file, class_file]

        coco_tiler.create_coco_from_photopea_layers(scene_id, layer_path, coco_output)

        os.remove(background_file)
        assert len(coco_tiler.coco_output["annotations"]) == 4
