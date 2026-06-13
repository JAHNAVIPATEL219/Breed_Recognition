import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("breed_model.h5")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping (index → class name)
class_names = {v: k for k, v in class_indices.items()}

# Load image
img_path = "test.jpg"   # change if needed
img = image.load_img(img_path, target_size=(224, 224))

# Preprocess image
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)

# Top 3 predictions
top3 = np.argsort(prediction[0])[-3:][::-1]

print("\n🔍 Top Predictions:")
for i in top3:
    class_name = class_names.get(i, "Unknown")
    confidence = prediction[0][i] * 100
    print(f"{class_name} : {confidence:.2f}%")