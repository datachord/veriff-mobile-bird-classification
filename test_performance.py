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


def test_performance():
    start_time = time.time()
    clf = BirdClassifier()
    for i, url in enumerate(image_urls):
        top_names, top_scores = clf.identify(url)
        # Print results to kubernetes log
        print(f'Run: {i + 1}')
        print(f'Top match: {top_names[0]} with score: {top_scores[0]}')
        print(f'Second match: {top_names[1]} with score: {top_scores[1]}')
        print(f'Third match: {top_names[2]} with score: {top_scores[2]}\n')
    print(f'Time spent: {time.time() - start_time}')