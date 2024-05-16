import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model('foodnutritio_baru.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict(img_path):
    data = request.get_json()
    image_url = data['image_url']

    # Assuming preprocessed_image_from_url is defined somewhere
    preprocessed_image = preprocessed_image_from_url(image_url)
    prediction = model.predict(preprocessed_image)

    # preds = model.predict(x)
    return f"prediction:{np.argmax(x[0])}"

if __name__ == "__main__":
    app.run(debug=True)