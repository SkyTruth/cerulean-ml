#!/usr/bin/env python

"""Tests for `ceruleanml` package."""

import pytest

from click.testing import CliRunner
import os

from ceruleanml import ceruleanml
from ceruleanml import data
from ceruleanml import cli

import skimage.io as skio
import numpy as np
import tempfile


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


def test_dist_array_from_tile(httpx_mock):
    httpx_mock.add_response(json=mock_scene_info())
    layer_path = [
        "tests/fixtures/S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5/cv2_transfer_outputs_skytruth_annotation_first_phase_old_vessel_S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5_ambiguous_1.png"
    ]
    arr = data.COCOtiler.dist_array_from_tile(
        layer_path,
        vector_ds="tests/fixtures/oil_areas_inverted_clip.geojson",
        resample_ratio=10,
    )
    assert arr.shape == (4181, 6458)
    assert arr.dtype == np.dtype(np.uint8)
    assert np.max(arr) == 255
    assert np.min(arr) == 0


def test_create_coco_from_photopea_layers(httpx_mock):
    httpx_mock.add_response(json=mock_scene_info())
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

        class_file = "tests/fixtures/S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5/cv2_transfer_outputs_skytruth_annotation_first_phase_old_vessel_S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5_ambiguous_1.png"
        template = skio.imread(class_file)
        background_file = class_file.replace("ambiguous_1", "Background")
        skio.imsave(background_file, template[:, :, 0])
        layer_path = [background_file, class_file]

        # Pass same vector dataset twice to make RGB image
        coco_tiler.save_background_img_tiles(
            layer_path,
            aux_datasets=[
                "tests/fixtures/oil_areas_inverted_clip.geojson",
                "tests/fixtures/oil_areas_inverted_clip.geojson",
            ],
            resample_ratio=10,
        )

        os.remove(background_file)
        assert len(os.listdir(tmp_dir)) == 117
