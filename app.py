from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Create Flask application
app = Flask(__name__)

# Folder where uploaded images will be stored
UPLOAD_FOLDER = 'static/uploads'

# Tell Flask where uploaded files should be saved
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't already exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained CIFAR-10 model
model = load_model('model/image_classification.h5', compile = False)

# CIFAR-10 class names
classes = {
    0: 'Airplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return render_template(
            'index.html',
            error='Please upload an image.'
        )

    file = request.files['image']

    if file.filename == '':
        return render_template(
            'index.html',
            error='Please select an image.'
        )

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        filename
    )

    file.save(filepath)

    image = Image.open(filepath)
    image = image.convert('RGB')
    image = image.resize((32, 32))

    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    predicted_class = np.argmax(
        prediction,
        axis=1
    )[0]

    result = classes[predicted_class]

    confidence = float(
        np.max(prediction)
    ) * 100

    return render_template(
        'index.html',
        prediction=result,
        confidence=round(confidence, 2),
        image_path=filename
    )

if __name__ == '__main__':
    app.run(debug=True)