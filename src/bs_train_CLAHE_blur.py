# USAGE
# python bs_train_CLAHE_blur.py

# import the necessary packages
import argparse
import numpy as np
import pandas as pd
import pickle

import cv2
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

print('Tensorflow: ', tf.__version__)
print('Opencv: ', cv2.__version__)


# Image preprocessing function
def apply_clahe_gb(img):
    grey_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grey_image = np.uint8(grey_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(grey_image)
    img_gaussian = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    img_gaussian = np.float32(img_gaussian)
    rgb_img = np.repeat(img_gaussian[..., np.newaxis], 3, -1)
    return rgb_img


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str,
                default="../dataset/", help="path to the dataset")
ap.add_argument("-c", "--csv", type=str,
                default="../bs_labels.csv", help="path to save csv")
args = vars(ap.parse_args())

# Parameters definition
MODELS_PATH = '../models/'
MODEL_NAME = 'bs_xception_model_clahe_gb'
NUM_CLASSES = 4
TARGET_SIZE = (299, 299)
BATCH_SIZE = 64
EPOCHS = 100

# Import image data
print('Loading image data...')
label_df = pd.read_csv(args["csv"])

train_df = label_df[label_df['subset'] == 'train']
val_df = label_df[label_df['subset'] == 'validation']
test_df = label_df[label_df['subset'] == 'test']

# Data augmentation
print('Data augmentation...')
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=25,
                                   zoom_range=0.1,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   brightness_range=[0.7, 1.3],
                                   horizontal_flip=True,
                                   preprocessing_function=apply_clahe_gb,
                                   fill_mode='nearest')

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                            preprocessing_function=apply_clahe_gb)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    directory=args["dataset"],
                                                    x_col='file_path',
                                                    y_col='label',
                                                    class_mode='categorical',
                                                    target_size=TARGET_SIZE,
                                                    seed=42,
                                                    batch_size=BATCH_SIZE)

val_generator = test_datagen.flow_from_dataframe(val_df,
                                                 directory=args["dataset"],
                                                 x_col='file_path',
                                                 y_col='label',
                                                 class_mode='categorical',
                                                 target_size=TARGET_SIZE,
                                                 shuffle=False,
                                                 batch_size=BATCH_SIZE)

# Model definition
print('Creating model...')
input_ = Input(shape=(299, 299, 3))

base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_tensor=input_,
)

base_model.trainable = False

for i in range(-6, 0):
    base_model.layers[i].trainable = True

custom_model = base_model.output
custom_model = GlobalAveragePooling2D()(custom_model)
custom_model = Dense(1024, activation='relu')(custom_model)
custom_model = Dropout(0.2)(custom_model)
custom_model = Dense(512, activation='relu')(custom_model)
custom_model = Dropout(0.2)(custom_model)
custom_model = Dense(128, activation='relu')(custom_model)
custom_model = Dropout(0.2)(custom_model)
custom_model = Dense(NUM_CLASSES, activation='softmax')(custom_model)

model = Model(base_model.input, custom_model)

# Fitting optimization
es = EarlyStopping(monitor='val_accuracy',
                   patience=15,
                   restore_best_weights=True)

mc = ModelCheckpoint(filepath=MODELS_PATH + MODEL_NAME + '.h5',
                     monitor='val_accuracy',
                     mode='max',
                     save_best_only=True)

lr_start = 0.001
lr_decay = 0.9
lrs = LearningRateScheduler(lambda epoch: lr_start * np.power(lr_decay, epoch))

# Fit model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Fitting model...')
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.n // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=val_generator,
                    validation_steps=val_generator.n // BATCH_SIZE,
                    callbacks=[es, mc, lrs])

# Save model history dictionary
print('Saving training history...')
with open(MODELS_PATH + MODEL_NAME + '_hist', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
