import pytest
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import os

# Constants
MODEL_PATH = "../model/emotion_model.h5"
IMG_PATH = "../images/sample_faces.png"

@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), "Model file not found."
    return load_model(MODEL_PATH)

def test_model_loaded(model):
    assert model is not None, "Model loading failed."

def test_input_output_shape(model):
    img = cv2.imread(IMG_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)

    prediction = model.predict(face)
    assert prediction.shape == (1, 7), "Unexpected prediction shape."
    assert np.isclose(np.sum(prediction), 1.0, atol=0.05), "Probabilities should sum to ~1."

def test_handles_blank_input(model):
    blank = np.zeros((1, 48, 48, 1))
    prediction = model.predict(blank)
    assert prediction.shape == (1, 7), "Model should return 7 emotion scores."
