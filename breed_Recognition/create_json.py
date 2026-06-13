from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Load dataset
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('data_split/train')

print("Classes found:", train_data.class_indices)

# Save class mapping
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=4)

print("✅ class_indices.json created successfully")