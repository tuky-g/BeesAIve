import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from PIL import Image
import os.path
from os import path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from skimage.transform import resize
import tensorflow as tf

app = Flask(__name__)

## MODEL'S PATH (change it to your specific model)
model = load_model('models/bs_xception_model.h5')

LAYER = 'block14_sepconv2_act'
TARGET_SIZE = (299, 299)
TARGET_CANALS_SIZE = (299, 299, 3)


@app.route("/", methods=['GET', 'POST'])
def main_page():
    remove_files('uploads')
    remove_files('static/grad_cam')

    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')


@app.route('/prediction/<filename>')
def prediction(filename):
    file_path = os.path.join('uploads', filename)

    my_image = plt.imread(file_path)
    my_image_re = resize(my_image, TARGET_CANALS_SIZE)

    probabilities = model.predict(np.array([my_image_re, ]))[0, :]
    probs_rounded = list(map(convert_probabilities, probabilities))

    number_to_class = ['bee', 'wasp', 'other thing', 'other insect']
    index = np.argsort(probs_rounded)
    predictions = {
        "class1": number_to_class[index[3]],
        "class2": number_to_class[index[2]],
        "class3": number_to_class[index[1]],
        "class4": number_to_class[index[0]],
        "prob1": probs_rounded[index[3]],
        "prob2": probs_rounded[index[2]],
        "prob3": probs_rounded[index[1]],
        "prob4": probs_rounded[index[0]],
    }

    processed_images= cam(file_path, model, LAYER)

    Image.fromarray(processed_images[0]).save(os.path.join('static/grad_cam', 'original_image.png'))
    Image.fromarray(processed_images[1]).save(os.path.join('static/grad_cam', 'heatmap_image.png'))
    Image.fromarray(processed_images[2]).save(os.path.join('static/grad_cam', 'superimposed_image.png'))
    Image.fromarray(processed_images[3]).save(os.path.join('static/grad_cam', 'rectangle_image.png'))

    return render_template('predict.html', predictions=predictions)


def convert_probabilities(prob):
    return round(prob * 100, 2)

def remove_files(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def cam(img_path, model, layer):
    # Importamos la imagen original
    img_o = cv2.imread(img_path)
    img_o = cv2.resize(img_o, TARGET_SIZE)
    img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
    img_c = img_o.copy()

    # Creamos el heatmap
    img_t = np.expand_dims(img_c / 255, axis=0)
    heatmap = make_gradcam_heatmap(img_t, model, layer)
    heatmap = np.uint8(255 * heatmap)

    # Le asignamos un mapa de color
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Creamos una imagen RGB con el heatmap coloreado
    jet_heatmap = cv2.resize(jet_heatmap, TARGET_SIZE)
    jet_heatmap = np.uint8(jet_heatmap * 255)

    # Superponemos el mapa de calor y la imagen original
    alpha = 0.4
    superimposed_img = cv2.addWeighted(jet_heatmap, alpha, img_o, 1 - alpha, 0)
    superimposed_img = np.uint8(superimposed_img)

    # Usamos el mapa de calor para obtener el recuadro
    grey_img = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey_img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    for item in range(len(contours)):
        cnt = contours[item]
        x, y, w, h = cv2.boundingRect(cnt)

    # Ponemos el recuadro sobre la imagen original
    rectangle = cv2.rectangle(img_o,
                              pt1=(x, y),
                              pt2=(x + w, y + h),
                              color=(255, 0, 0),
                              thickness=2)

    image_type = [img_c, jet_heatmap, superimposed_img, rectangle]

    return image_type


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()




if __name__ == "__main__":
    app.run(host="localhost", port=8082, debug=True)