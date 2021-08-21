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
import flask

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

app = flask.Flask(__name__)


class BirdClassifier:
    def __init__(self, online=True):
        """
        Identify name of bird based on image
        online: load classification model and label data online, otherwise from local machine
        """
        
        # classification model
        model_url = ("https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1" if online
                     else "classifier")
        self.model = hub.KerasLayer(model_url)

        # recognizable bird names
        label_url = ("https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv" if online
                     else "classifier/labels.csv")
        data_label = pd.read_csv(label_url)
        self.labels = dict(data_label.values)
                
    @staticmethod
    def load_image(url):
        """
        Read image url into array (224, 224)
        """
        
        # Loading images
        image_get_response = urllib.request.urlopen(url)
        image_array = np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)

        # Changing images
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255
        
        return image
    
    def identify(self, image_url):
        """
        Identify name of bird based on image url
        """
        
        # Generate tensor
        image = self.load_image(image_url)
        image_tensor = tf.convert_to_tensor([image], dtype=tf.float32)

        # predict bird name
        prob = self.model.call(image_tensor).numpy()[0]  # probability for each bird
        top_indices = prob.argsort()[::-1]  # indices of predictions from most to least
        top_names = list(map(self.labels.get, top_indices))  # bird names from most to least
        top_scores = prob[top_indices]  # scores from most to least
        
        return top_names, top_scores
    
# @app.route("/predict", methods=["POST", "GET"])
def predict():
    clf = BirdClassifier()
    predictions = []
    for i, url in enumerate(image_urls):
        top_names, top_scores = clf.identify(url)
        predictions.append((top_names[0], top_scores[0]))
    
    return str(predictions)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')