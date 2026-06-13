from flask import Flask, render_template, request
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ✅ Absolute path fix (IMPORTANT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create static folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model("breed_model.h5")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping
class_names = {v: k for k, v in class_indices.items()}


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    img_path = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filename = file.filename

            # Save image safely
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Path for HTML display
            img_path = f"static/{filename}"

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            pred = model.predict(img_array)

            # Top 3 predictions
            top3 = np.argsort(pred[0])[-3:][::-1]

            predictions = []
            for i in top3:
                predictions.append({
                    "class": class_names.get(i, "Unknown"),
                    "confidence": round(float(pred[0][i]) * 100, 2)
                })

    return render_template("index.html", predictions=predictions, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)