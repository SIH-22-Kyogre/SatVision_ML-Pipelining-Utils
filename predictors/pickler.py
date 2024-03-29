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
    "eurosat.h5",
    ""
)

PREDICTOR_NAME = "predict.pkl"
# SET: predictor name (default: predict.pkl)
PREDICTOR_NAME = "efficientnetv0.pkl"

# SET: model to load
model = make_vgg16_classifier()

# --------------------------------

model.summary()
print("Model loaded")

model.load_weights(WEIGHT_FILEPATH)
print(f"Weights loaded from {WEIGHT_FILEPATH}")

predictor_filepath = os.path.join(
    os.path.split(WEIGHT_FILEPATH)[0],  # get container
    PREDICTOR_NAME
)
pickle.dump(
    model,
    open(predictor_filepath, 'wb')
)
print(f"Pickled to {predictor_filepath}")


# EfficientNetV0 - bin to pkl, hopefully
# comment if not commented in Debian

# PATH = r"models\\model4.bin"

# model = torch.load(PATH)
# # model.eval()

# with open(r"models\\model4.pkl","wb") as f:
#     pickle.dump(model, f)

