import os
import numpy as np
import cv2
import tensorflow as tf
import firebase_admin
from skimage.metrics import structural_similarity as ssim
from flask import Flask, render_template, request
from firebase_admin import credentials, storage

# Initialize Firebase
cred = credentials.Certificate("app/firebase_config.json")
firebase_admin.initialize_app(cred, {"storageBucket": "your-project-id.appspot.com"})
bucket = storage.bucket()

# Load Pre-trained Model
model = tf.keras.models.load_model("app/model/fake_logo_model.h5")

# Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")
UPLOAD_FOLDER = "app/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to Predict Fake or Real Logo
def predict_logo(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)[0][0]
    return "Fake Logo" if prediction > 0.5 else "Real Logo"

# Function to Compute SSIM Similarity
def compute_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))
    
    score, _ = ssim(img1, img2, full=True)
    return round(score * 100, 2)  # Convert to percentage

# Function to Upload Image to Firebase and Get URL
def upload_to_firebase(image_path):
    blob = bucket.blob(f"logos/{os.path.basename(image_path)}")
    blob.upload_from_filename(image_path)
    blob.make_public()
    return blob.public_url

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file1" not in request.files or "file2" not in request.files:
        return "Both files are required", 400

    file1 = request.files["file1"]
    file2 = request.files["file2"]

    if file1.filename == "" or file2.filename == "":
        return "Both files must be selected", 400

    file1_path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
    file2_path = os.path.join(app.config["UPLOAD_FOLDER"], file2.filename)
    
    file1.save(file1_path)
    file2.save(file2_path)

    # Predict Fake/Real for both logos
    result1 = predict_logo(file1_path)
    result2 = predict_logo(file2_path)

    # Calculate SSIM Similarity
    similarity = compute_ssim(file1_path, file2_path)

    # Upload images to Firebase
    image1_url = upload_to_firebase(file1_path)
    image2_url = upload_to_firebase(file2_path)

    return render_template(
        "result.html", 
        result1=result1, image1_url=image1_url, 
        result2=result2, image2_url=image2_url, 
        similarity=similarity
    )

if __name__ == "__main__":
    app.run(debug=True)
