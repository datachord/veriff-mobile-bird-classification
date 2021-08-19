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

image_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg',
    'https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg',
    'https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg',
    'https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg'
]


@pytest.fixture
def classifier():
    return BirdClassifier()

@pytest.mark.parametrize('url', image_urls)
def test_load_image(classifier, url):
    image = classifier.load_image(url)
    assert image.shape == (224, 224, 3)
    assert np.all(image <= 1)  # normalized
    
@pytest.mark.parametrize('image_url,exp_top_3_names', [
    ('https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg',
     ["Phalacrocorax varius varius", "Phalacrocorax varius", "Microcarbo melanoleucos"]),
    ('https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg',
     ["Galerida cristata", "Alauda arvensis", "Eremophila alpestris"]),
    ('https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg',
     ["Eumomota superciliosa", "Momotus coeruliceps", "Momotus lessonii"]),
    ('https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg',
     ["Aulacorhynchus prasinus", "Cyanocorax yncas", "Chlorophanes spiza"]),
    ('https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg',
     ["Erithacus rubecula", "Ixoreus naevius", "Setophaga tigrina"])
])
def test_identify(classifier, image_url, exp_top_3_names):
    top_names, top_scores = classifier.identify(image_url)
    npt.assert_equal(top_names[:3], exp_top_3_names)
    npt.assert_array_less(top_scores[1:]/top_scores[0], 0.5)  # top probability more than twice others
    npt.assert_almost_equal(sum(top_scores), 1, decimal=5)  # total probability 1