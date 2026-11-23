
from flask import Flask, render_template, request, send_from_directory
import os
import tensorflow as tf
import numpy as np

from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg19 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained VGG19 model
model = tf.keras.models.load_model(
    "vgg19_deepfake_model.keras"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:
        return "No image selected!"

    file = request.files["image"]

    if file.filename == "":
        return "Please select an image!"

    filename = secure_filename(file.filename)

    if not filename:
        return "Invalid filename!"

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(file_path)

    # Load and resize image
    img = tf.keras.utils.load_img(
        file_path,
        target_size=(224, 224)
    )

    # Convert image into array
    img_array = tf.keras.utils.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Apply VGG19 preprocessing
    img_array = preprocess_input(img_array)

    # Predict image
    prediction = model.predict(img_array)

    score = float(prediction[0][0])

    # Classify result and calculate confidence
    if score >= 0.5:
        result = "REAL"
        confidence = score * 100
    else:
        result = "FAKE"
        confidence = (1 - score) * 100

    confidence = round(confidence, 2)

    # Display prediction result
    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        score=score,
        filename=filename
    )


if __name__ == "__main__":
    app.run(debug=True)
