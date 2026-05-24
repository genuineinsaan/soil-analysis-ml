import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# DATASET PATH
# -----------------------------

dataset_path = r"dataset\soil_images\Original-Dataset"

IMG_SIZE = 224

# -----------------------------
# LOAD DATASET
# -----------------------------

data = []
labels = []

classes = os.listdir(dataset_path)

print("Classes Found:")
print(classes)

for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        try:

            image = cv2.imread(image_path)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            # MobileNet preprocessing
            image = preprocess_input(image)

            data.append(image)

            labels.append(class_name)

        except Exception as e:

            print(f"Error loading image: {image_path}")

# Convert arrays
data = np.array(data)
labels = np.array(labels)

# -----------------------------
# ENCODE LABELS
# -----------------------------

label_encoder = LabelEncoder()

labels_encoded = label_encoder.fit_transform(labels)

labels_categorical = to_categorical(labels_encoded)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels_categorical,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

print("\nDataset Loaded Successfully!")

print("Training Images:", len(X_train))
print("Testing Images:", len(X_test))

# -----------------------------
# CLASS WEIGHTS
# -----------------------------

class_weights = compute_class_weight(

    class_weight='balanced',

    classes=np.unique(labels_encoded),

    y=labels_encoded
)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights:")
print(class_weights)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------

datagen = ImageDataGenerator(

    rotation_range=25,

    zoom_range=0.25,

    horizontal_flip=True,

    width_shift_range=0.25,

    height_shift_range=0.25,

    brightness_range=[0.8, 1.2]
)

datagen.fit(X_train)

# -----------------------------
# LOAD MOBILENETV2
# -----------------------------

base_model = MobileNetV2(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# -----------------------------
# BUILD MODEL
# -----------------------------

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(len(classes), activation='softmax')
])

# -----------------------------
# COMPILE MODEL
# -----------------------------

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

print("\nTraining Started...\n")

# -----------------------------
# TRAIN MODEL
# -----------------------------

history = model.fit(

    datagen.flow(X_train, y_train, batch_size=32),

    epochs=20,

    validation_data=(X_test, y_test),

    class_weight=class_weights
)

# -----------------------------
# EVALUATE MODEL
# -----------------------------

loss, accuracy = model.evaluate(X_test, y_test)

print("\nFinal Model Accuracy:", accuracy * 100)

# -----------------------------
# SAVE MODEL
# -----------------------------

model.save("models/soil_cnn_model.h5")

print("\nFinal Improved Model Saved Successfully!")