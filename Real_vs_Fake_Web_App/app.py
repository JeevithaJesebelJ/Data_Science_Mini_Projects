import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Load trained model
model = load_model("real_vs_fake_final.keras")


def predict_image(img_path):

    # Open image and convert to grayscale
    img = Image.open(img_path).convert("L")

    # Resize exactly as used during training
    img = img.resize((64, 64))

    # Convert image to numpy array
    img_array = np.array(img)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Add channel dimension
    img_array = np.expand_dims(img_array, axis=-1)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    # Class mapping
    # fake = 0
    # real = 1

    if prediction >= 0.5:
        result = "REAL"
        confidence = prediction * 100
    else:
        result = "FAKE"
        confidence = (1 - prediction) * 100

    return result, confidence


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    confidence = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "index.html",
                result="No image uploaded"
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                result="Please select an image"
            )

        upload_folder = "uploads"

        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        filepath = os.path.join(upload_folder, file.filename)

        file.save(filepath)

        result, confidence = predict_image(filepath)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)