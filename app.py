from flask import Flask, request, jsonify
import werkzeug
import firebase_admin
from firebase_admin import credentials, storage
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from joblib import load
import pandas as pd

from rembg import remove

import jwt 
from functools import wraps

app = Flask(__name__)

cert_path = 'firebase/creds.json'

if os.path.exists(cert_path):
    cred = credentials.Certificate(cert_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'capstoneproject-ch2ps217.appspot.com'
    })
else:
    print(f"File '{cert_path}' does not exist.")

knn_model = load_model('model/knn_model.h5')

model_path = "model/trained_model.h5"
trained_model = load_model(model_path)

class_labels_color = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'red', 'violet', 'white', 'yellow']
class_labels_image = ["Accessories", "Bottomwear", "Dress", "Sandals", "Shoes", "Topwear"]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_color_from_bytes(input_image_bytes, knn_model, class_labels):
    image_np = np.frombuffer(input_image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image_rgb, image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    resized_rgb, resized_hsv = cv2.resize(image_rgb, (1, 1)), cv2.resize(image_hsv, (1, 1))

    color_data_rgb, color_data_hsv = resized_rgb.reshape(-1, 3), resized_hsv.reshape(-1, 3)
    combined_data = np.hstack((color_data_rgb, color_data_hsv))

    predictions = knn_model.predict(combined_data)
    predicted_color_index = np.argmax(predictions[0])

    predicted_color = class_labels[predicted_color_index]

    return predicted_color

def predict_image_class_from_bytes(input_image_bytes):
    img_array = preprocess_image_from_bytes(input_image_bytes)
    predictions = trained_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    return predicted_class_index, predictions

def preprocess_image_from_bytes(input_image_bytes):
    image_np = np.frombuffer(input_image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    return img_array

def remove_background(input_image):
    output_image = remove(input_image, alpha_matting=True, alpha_matting_background_threshold=50)
    return output_image

#buat token 
SECRET_KEY = 'pwf1rebase0!'
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        if token.startswith('Bearer '):
            token = token.split(' ')[1]

        print("Received token:", token)    

        try:
            # Decode the token using the secret key
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            uid = decoded_token['uid']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401

        # Pass the UID to the route function using **kwargs
        return f(uid, *args, **kwargs)
    
    return decorated

@app.route("/")
def index():
    return jsonify({
        "status":{
            "code": 200,
            "message": "Success"
            },
            "data": "Hello World"
        }), 200

@app.route('/upload', methods=['POST'])
@token_required
def upload_file(uid):
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'}), 400

    if file:
        filename = werkzeug.utils.secure_filename(file.filename)

        with file.stream as file_stream:
            input_image = file_stream.read()
        output_image = remove_background(input_image)

        color_prediction = predict_color_from_bytes(output_image, knn_model, class_labels_color)
        predicted_class_index, _ = predict_image_class_from_bytes(output_image)
        predicted_class = class_labels_image[predicted_class_index]

        bucket = storage.bucket()
        final_output_path = f"userimages/{uid}/clothes/{filename[:-4]}_{predicted_class}_{color_prediction}.jpg"  # Adjust the naming convention
        blob = bucket.blob(final_output_path)

        setMetadata = {
            'category': predicted_class,
            'color': color_prediction
        }

        blob.metadata = setMetadata
        blob.upload_from_string(output_image, content_type='image/jpeg')

        return jsonify({'message': 'File successfully uploaded and processed', 'predicted_class': predicted_class, 'color': color_prediction}), 200

    return jsonify({'message': 'Something went wrong'}), 500

if __name__ == '__main__':
    app.run(debug=True)
