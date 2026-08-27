from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os


app = Flask(__name__)

# Load trained models
age_model = load_model("age_model.keras")
gender_model = load_model("gender_model.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    # Save uploaded image
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    image_path = os.path.join(upload_folder, file.filename)
    file.save(image_path)

    # Read image
    img = cv2.imread(image_path)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert image
    img = 255 - img

    # Resize
    img = cv2.resize(img, (128, 128))

    # Reshape
    img = img.reshape(1, 128, 128, 1)

    # Normalize
    img = img / 255.0

    # Predict age
    age_prediction = age_model.predict(img, verbose=0)

    # Predict gender
    gender_prediction = gender_model.predict(img, verbose=0)

    # Convert age prediction
    predicted_age = round(float(age_prediction[0][0]))

    # Convert gender prediction
    if gender_prediction[0][0] < 0.5:
        predicted_gender = "Female"
    else:
        predicted_gender = "Male"

    prediction = {
        "age": predicted_age,
        "gender": predicted_gender
    }

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)