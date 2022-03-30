#!/usr/bin/env python

"""Tests for `ceruleanml` package."""

import pytest

from click.testing import CliRunner

from ceruleanml import ceruleanml
from ceruleanml import data
from ceruleanml import cli


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
    assert 'ceruleanml.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_dist_array_from_tile():
    layer_path = ["tests/fixtures/S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5/cv2_transfer_outputs_skytruth_annotation_first_phase_old_vessel_S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5_ambiguous_1.png"]
    data.COCOtiler.dist_array_from_tile(layer_path)


def test_create_coco_from_photopea_layers():
    info = {
        "description": "Cerulean Dataset V2",
        "url": "none",
        "version": "1.0",
        "year": 2021,
        "contributor": "Skytruth",
        "date_created": "2022/2/23"
    }

    licenses = [
        {
            "url": "none",
            "id": 1,
            "name": "CeruleanDataset V2"
        }
    ]
    categories = [{"supercategory": "slick", "id": 1, "name": "infra_slick"},
                  {"supercategory": "slick", "id": 2, "name": "natural_seep"},
                  {"supercategory": "slick", "id": 3, "name": "coincident_vessel"},
                  {"supercategory": "slick", "id": 4, "name": "recent_vessel"},
                  {"supercategory": "slick", "id": 5, "name": "old_vessel"},
                  {"supercategory": "slick", "id": 6, "name": "ambiguous"}]

    coco_output = {
        "info": info,
        "licenses": licenses,
        "images": [],
        "annotations": [],
        "categories": categories
    }
    coco_tiler = data.COCOtiler("tiled_images/", coco_output)
    layer_path = ["tests/fixtures/S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5/cv2_transfer_outputs_skytruth_annotation_first_phase_old_vessel_S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5_Background.png",
                  "tests/fixtures/S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5/cv2_transfer_outputs_skytruth_annotation_first_phase_old_vessel_S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5_ambiguous_1.png"]

    coco_tiler.save_background_img_tiles(layer_path, aux_datasets=["/Users/rodrigoalmeida/cerulean-ml/inverted.json"])
