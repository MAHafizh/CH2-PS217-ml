from flask import Flask, request, jsonify, g, send_file
import werkzeug
import firebase_admin
from firebase_admin import credentials, storage
import os
from io import BytesIO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from joblib import load
import pandas as pd
import requests
import tempfile
from PIL import Image
import shutil
from itertools import product

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

def find_matching_items(category, preferred_colors, all_items):
    # Find items that match the category and are of the preferred colors
    matching_items = [item for item in all_items if item['category'] == category]
    color_matched_items = [item for item in matching_items if item['color'] in preferred_colors]

    return color_matched_items if color_matched_items else matching_items

def process_image_and_generate_outfits(selected_image, all_images):

    _, selected_color = parse_filename(os.path.basename(selected_image))

    # Extract the color part from the filename
    _, color_part = selected_color.split('_', 1)

    # Extract only the color value (assuming it's the last part)
    color_value = color_part.split('_')[-1]

    print("Selected color value in the function:", color_value)

    # Membuat kombinasi pakaian
    generate_all_outfit_combinations(color_value, all_images)

def generate_all_outfit_combinations(selected_color, all_images):
    outfit_categories = ['Topwear', 'Bottomwear', 'Shoes', 'Accessories']
    outfits = []

    # Find matching items for each category
    matches_by_category = {
        category: find_matching_items(category, [selected_color], all_images)
        for category in outfit_categories
    }

    # Menampilkan dan mengakses nama-nama gambar berdasarkan kategori
    for category, items in matches_by_category.items():
        print(f"\n{category} Items:")
        for item in items:
            filename = item['filename']
            print(filename)

            # Anda juga dapat menyimpan nama file ke dalam list atau variabel lain
            # misalnya, outfits.append(filename) jika Anda ingin menyimpannya untuk penggunaan lebih lanjut

    # save_outfits_to_local_folder(outfits)

def save_outfits_to_local_folder(outfits):
    outfit_path = "output_outfit"
    os.makedirs(outfit_path, exist_ok=True)
    print("Saving Recommended Outfits to Local Folder", outfit_path)

    # Save only the first two outfits for simplicity
    for outfit_index, outfit in enumerate(outfits[:2]):
        outfit_folder = os.path.join(outfit_path, f"outfit_{outfit_index + 1}")
        os.makedirs(outfit_folder, exist_ok=True)

        for i, (category, filename) in enumerate(outfit.items(), 1):
            img = cv2.imread(os.path.join(outfit_path, filename))

            # Save the image to the local outfit folder
            output_path = os.path.join(outfit_folder, f"{category}_{filename}")
            cv2.imwrite(output_path, img)

    print("Outfits saved to local folder.")

def parse_filename(filename):
    # Assuming the filename format is: "name-category-color.jpg"
    parts = filename.split('-')
    if len(parts) < 3:
        return None, None
    return parts[-3], parts[-2], parts[-1].split('.')[0]  # Extract category and color

def get_selected_image(uid, filename):
    try:
        if not uid:
            return None
        
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix=f"userimages/{uid}/clothes/")

        selected_image = None

        for blob in blobs:
            filesname = blob.name
            file_name_ext = filesname.split("/")[-1]
            file_name = file_name_ext

            os.makedirs("output_folder", exist_ok=True)

            url = f"https://storage.googleapis.com/{bucket.name}/{blob.name}"
            file_name_without_extension, file_extension = os.path.splitext(file_name)
            image_output = os.path.join("output_folder", f"{file_name_without_extension}.png")
            download_image(url, image_output) 

            if file_name == filename:
                selected_image = os.path.join("output_folder", f"{file_name_without_extension}.png")
                break

        return selected_image 
    except Exception as error:
        print(f"error: {str(error)}")
        return None

def process_images(output_folder):
    all_images=[]
    # Scan the output images folder
    for filename in os.listdir(output_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            category, color = parse_filename(filename)
            if category and color:
                all_images.append({'filename': filename, 'category': category, 'color': color})

    return all_images

def download_image(url, local_filename):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(local_filename, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded successfully to {local_filename}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

def upload_file_to_storage(local_file_path, remote_file_path):
    try:
        # Membuat instance dari Firebase Storage
        storage_client = storage.bucket()

        # Mengunggah file ke Firebase Storage
        blob = storage_client.blob(remote_file_path)
        blob.upload_from_filename(local_file_path)

        print(f"File {local_file_path} berhasil diunggah ke {remote_file_path}")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def create_outfits_folder(uid):
    try:
        bucket = storage.bucket()
        folder_name = f"userimages/{uid}/outfits/"

        # Mengecek apakah folder outfits sudah ada
        blobs = bucket.list_blobs(prefix=folder_name)
        if not any(blobs):
            blob = bucket.blob(folder_name)
            blob.upload_from_string('')  # Membuat folder outfits

        # Mencari nama folder yang belum terpakai
        count = 1
        while True:
            folder_name_with_count = f"userimages/{uid}/outfits/myoutfits_{count}"
            blobs = bucket.list_blobs(prefix=folder_name_with_count)
            if not any (blobs):
                blob = bucket.blob(folder_name_with_count)
                blob.upload_from_string('')
                break 

            count += 1

        return folder_name_with_count

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

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

#############################################################################################################

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
def upload_image(uid):
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'}), 400

    if file:
        filename = werkzeug.utils.secure_filename(file.filename)

        with file.stream as file_stream:
            input_image = file_stream.read()
        output_image = remove_background(input_image)

        g.filename = filename

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
        url = f"https://storage.googleapis.com/{bucket.name}/{final_output_path}"
        g.url = url

        return jsonify({
            'message': 'File successfully uploaded and processed',
            'url': url,
            'predicted_class': predicted_class, 
            'color': color_prediction,
        }), 200

    return jsonify({'message': 'Something went wrong'}), 500

@app.route('/mix-match', methods=['GET'])
@token_required
def get_image(uid):
    try:
        user_filename = request.args.get('filename')

        selected_image = get_selected_image(uid, user_filename)
        selected_image = selected_image.replace("\\", "/")
        local_folder_path = "output_folder"

        all_images = process_images(local_folder_path)
        print("selected image:", selected_image)
        process_image_and_generate_outfits(user_filename, all_images)

        return jsonify({
            "status": {"code": 200, "message": "Success"},
            "data": {"message": "Image processing and outfit generation completed successfully"}
        }), 200

    except Exception as error:
        return jsonify({
            "status": {"code": 500, "message": "Internal Server Error"},
            "data": {"error": str(error)}
        }), 500
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)