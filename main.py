import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os

import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from skimage.transform import resize

app = Flask(__name__)

model = load_model('models/bs_resnet50.h5')


@app.route("/", methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')


@app.route('/prediction/<filename>')
def prediction(filename):
    # Step 1
    my_image = plt.imread(os.path.join('uploads', filename))
    # Step 2
    my_image_re = resize(my_image, (256, 256, 3))

    # Step 3
    probabilities = model.predict(np.array([my_image_re, ]))[0, :]
    print(probabilities)
    # Step 4
    number_to_class = ['bees', 'wasp', 'other insects', 'other']
    index = np.argsort(probabilities)
    predictions = {
        "class1": number_to_class[index[3]],
        "class2": number_to_class[index[2]],
        "class3": number_to_class[index[1]],
        "class4": number_to_class[index[0]],
        "prob1": probabilities[index[3]],
        "prob2": probabilities[index[2]],
        "prob3": probabilities[index[1]],
        "prob4": probabilities[index[0]],
    }
    # Step 5
    return render_template('predict.html', predictions=predictions)


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)
