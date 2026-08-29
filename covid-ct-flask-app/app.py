from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = load_model("CovidTest.h5")


def preprocess_image(image_path):
    
    # Open image
    image = Image.open(image_path)

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Resize image
    size = (128, 128)
    image = ImageOps.fit(
        image,
        size,
        Image.Resampling.LANCZOS
    )

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Reshape and normalize
    data = image_array.reshape(
        (-1, 128, 128, 1)
    ) / 255.0

    return data


@app.route("/", methods=["GET", "POST"])
def index():

    prediction_text = None
    uploaded_image = None

    if request.method == "POST":

        file = request.files.get("file")

        if file and file.filename != "":

            # Create uploads folder if it doesn't exist
            os.makedirs(
                app.config["UPLOAD_FOLDER"],
                exist_ok=True
            )

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            # Preprocess image
            image_data = preprocess_image(filepath)

            # Predict
            prediction = model.predict(image_data)[0][0]

            # Classification
            if prediction < 0.5:
                prediction_text = "COVID Detected"
            else:
                prediction_text = "No COVID Detected"

            uploaded_image = filepath

    return render_template(
        "index.html",
        prediction=prediction_text,
        image=uploaded_image
    )


if __name__ == "__main__":
    app.run(debug=True)