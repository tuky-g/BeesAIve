# USAGE
# python bs_predict_evaluation.py

# import the necessary packages
import argparse
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str,
                default="../dataset/", help="path to the dataset")
ap.add_argument("-c", "--csv", type=str,
                default="../bs_labels.csv", help="path to save csv")
args = vars(ap.parse_args())

# Define parameters
MODELS_PATH = '../models/'
MODEL_NAME = 'bs_xception_model'
REPORTS_PATH = '../reports/'
NUM_CLASSES = 4
TARGET_SIZE = (299, 299)
BATCH_SIZE = 64

# Get test subset
print('Loading image data...')
label_df = pd.read_csv(args["csv"])
test_df = label_df[label_df['subset'] == 'test']

# Preprocess test data
print('Preprocess test data...')
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_dataframe(test_df,
                                                  directory=args["dataset"],
                                                  x_col='file_path',
                                                  y_col='label',
                                                  class_mode='categorical',
                                                  target_size=TARGET_SIZE,
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# Evaluate model
print('Load model...')
model = load_model(MODELS_PATH + MODEL_NAME + '.h5')

print('Calculate test accuracy...')
start_time = time.time()
test_loss, test_acc = model.evaluate(test_generator,
                                     steps=test_generator.n // BATCH_SIZE,
                                     verbose=2)

print("Test accuracy:", test_acc)

ej_time = time.time() - start_time
time_by_batch = ej_time / (test_generator.n // BATCH_SIZE)
print("Prediction time by batch:", time_by_batch)

# Get predictions from test set
print('Make predictions...')
predictions = model.predict(test_generator)
top_score = np.argmax(predictions, axis=1)

print('Calculate confusion matrix...')
# Calculate confusion matrix
matrix = confusion_matrix(test_generator.classes, top_score)

# Show confusion matrix
plt.figure(figsize=(10, 8))
class_labels = ['bee', 'other_insect', 'other_noinsect', 'wasp']
df_cm = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
plt.yticks(va="center")
sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels', fontsize=14)
plt.ylabel('True labels', fontsize=14)
plt.show()

# Show classification report
print('Generate classification report...')
print(classification_report(test_generator.classes,
                            top_score,
                            target_names=class_labels))

print('Save classification report...')
report_dict = classification_report(test_generator.classes,
                                    top_score,
                                    target_names=class_labels,
                                    output_dict=True)
df_r = pd.DataFrame(report_dict).transpose().reset_index()
df_r['ej_time'] = time_by_batch
df_r.to_csv(REPORTS_PATH + MODEL_NAME + '_report.csv', index=False)
