import pickle
import os

# TODO: Write a recursively parsed pipeliner defn

from model import *

# SET: weight file path
WEIGHT_FILEPATH = os.path.join(
    os.path.abspath(os.path.join(__file__, os.path.pardir)),
    os.path.pardir,
    "models",
    "vgg16",
    "eurosat.h5"
)

PREDICTOR_NAME = "predict.pkl"
# SET: predictor name (default: predict.pkl)
PREDICTOR_NAME = "vgg16_classifier_eurosat.pkl"

# SET: model to load
model = get_vgg16_classifier()

# --------------------------------

model.summary()
print("Model loaded")

model.load_weights(WEIGHT_FILEPATH)
print(f"Weights loaded from {WEIGHT_FILEPATH}")

predictor_filepath = os.path.join(
    os.path.abspath(os.path.join(__file__, os.path.pardir)),
    PREDICTOR_NAME
)
pickle.dump(
    open(predictor_filepath, 'wb'),
    model
)
print(f"Pickled to {predictor_filepath}")