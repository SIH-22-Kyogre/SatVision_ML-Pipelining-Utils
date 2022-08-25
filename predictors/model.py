import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications.vgg16 import VGG16

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

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

# def make_efficientv0_classifier():

#     config = {'lr':5e-5,
#           'wd':1e-2,
#           'bs':64,
#           'img_size':256,
#           'nfolds':5,
#           'epochs':20,
#           'num_workers':4,
#           'seed':1000,
#           'model_name':'tf_efficientnet_b0',
#          }

#     classes = ['AnnualCrop', 'HerbaceousVegetation', 'PermanentCrop',
#        'Industrial', 'Pasture', 'Highway', 'Residential', 'River',
#        'SeaLake', 'Forest']

#     num_classes = len(classes)


#     class Model(nn.Module):
#         def __init__(self,model_path,pretrained=True):
#             super(Model,self).__init__()
#             self.backbone = timm.create_model(model_path,pretrained=pretrained)
#             in_features = self.backbone.classifier.in_features
#             self.backbone.classifier = nn.Linear(in_features,128)
#             self.dropout = nn.Dropout(0.2)
#             self.relu = nn.ReLU()
#             self.layer = nn.Linear(128, num_classes)

#         def forward(self,x):
#             x = self.relu(self.backbone(x))
#             x = self.layer(self.dropout(x))
#             return x
    
#     model = Model(config['model_name'])

#     return model
