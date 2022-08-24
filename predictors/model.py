import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications.vgg16 import VGG16

# Parameters
INPUT_SHAPE = (64, 64, 3)
LABELS = ['SeaLake', 'Highway', 'River', 'Pasture', 'Industrial', 'Residential', 'PermanentCrop', 'AnnualCrop', 'Forest', 'HerbaceousVegetation']
N_LABELS = len(LABELS)


def make_vgg16_classifier():
    
    conv_base = VGG16(
        include_top=False,
        weights='imagenet', 
        input_shape=INPUT_SHAPE
    )

    # Construct downstream layers
    top_model = conv_base.output
    top_model = Flatten()(top_model)
    top_model = Dense(2048, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(2048, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)   
    
    output_layer = Dense(N_LABELS, activation='softmax')(top_model)
    model = Model(inputs=conv_base.input, outputs=output_layer)
    return model
