#!/usr/bin/env python3

import os
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import cv2
import urllib.request
import pandas as pd
import numpy as np
import time
import pytest
import numpy.testing as npt

from classifier import BirdClassifier

# Getting some unknown linter errors, disable everything to get this to production asap
# pylint: disable-all

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow logging


@pytest.fixture
def classifier_offline():
    return BirdClassifier(online=False)

local_dir = "file:///work/veriff-mobile-bird-classification"  # repository directory – change /kaggle/working to suit local file system
@pytest.mark.parametrize('url', [
    f"{local_dir}/images/Phalacrocorax varius varius.jpg",
    f"{local_dir}/images/Galerida cristata.jpg",
    f"{local_dir}/images/Eumomota superciliosa.jpg",
    f"{local_dir}/images/Aulacorhynchus prasinus.jpg",
    f"{local_dir}/images/Erithacus rubecula.jpg"
])
def test_load_image_offline(classifier_offline, url):
    """
    url: mocked by local file path
    """
    image = classifier_offline.load_image(url)
    assert image.shape == (224, 224, 3)
    assert np.all(image <= 1)  # normalized
    
@pytest.mark.parametrize('image_url,exp_top_3_names', [
    (f'{local_dir}/images/Phalacrocorax varius varius.jpg',
     ["Phalacrocorax varius varius", "Phalacrocorax varius", "Microcarbo melanoleucos"]),
    (f'{local_dir}/images/Galerida cristata.jpg',
     ["Galerida cristata", "Alauda arvensis", "Eremophila alpestris"]),
    (f'{local_dir}/images/Eumomota superciliosa.jpg',
     ["Eumomota superciliosa", "Momotus coeruliceps", "Momotus lessonii"]),
    (f'{local_dir}/images/Aulacorhynchus prasinus.jpg',
     ["Aulacorhynchus prasinus", "Cyanocorax yncas", "Chlorophanes spiza"]),
    (f'{local_dir}/images/Erithacus rubecula.jpg',
     ["Erithacus rubecula", "Ixoreus naevius", "Setophaga tigrina"])
])
def test_identify_offline(classifier_offline, image_url, exp_top_3_names):
    """
    image_url: mocked by local image file path
    """
    top_names, top_scores = classifier_offline.identify(image_url)
    npt.assert_equal(top_names[:3], exp_top_3_names)
    npt.assert_array_less(top_scores[1:]/top_scores[0], 0.5)  # top probability more than twice others
    npt.assert_almost_equal(sum(top_scores), 1, decimal=5)  # total probability 1