import joblib
import pandas as pd
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# LOAD MODELS
# -----------------------------

# CNN model
model = load_model("models/soil_cnn_model.h5")

# Random Forest model
rf_model = joblib.load("models/random_forest_model.pkl")

# -----------------------------
# SOIL CLASS NAMES
# -----------------------------

class_names = [
    "Alluvial Soil",
    "Arid Soil",
    "Black Soil",
    "Laterite Soil",
    "Mountain Soil",
    "Red Soil",
    "Yellow Soil"
]

# -----------------------------
# SOIL INFORMATION
# -----------------------------

soil_info = {

    "Alluvial Soil": {
        "moisture": "Moderate",
        "crops": ["Rice", "Wheat", "Sugarcane"],
        "health_score": 85,
        "suggestion": "Maintain organic compost regularly."
    },

    "Arid Soil": {
        "moisture": "Low",
        "crops": ["Millets", "Barley"],
        "health_score": 52,
        "suggestion": "Increase irrigation and organic matter."
    },

    "Black Soil": {
        "moisture": "High",
        "crops": ["Cotton", "Soybean", "Wheat"],
        "health_score": 88,
        "suggestion": "Good soil quality. Maintain drainage properly."
    },

    "Laterite Soil": {
        "moisture": "Moderate",
        "crops": ["Tea", "Coffee", "Cashew"],
        "health_score": 68,
        "suggestion": "Add fertilizers for better productivity."
    },

    "Mountain Soil": {
        "moisture": "High",
        "crops": ["Tea", "Spices"],
        "health_score": 74,
        "suggestion": "Prevent soil erosion using vegetation."
    },

    "Red Soil": {
        "moisture": "Low",
        "crops": ["Groundnut", "Millets"],
        "health_score": 64,
        "suggestion": "Improve nitrogen content using compost."
    },

    "Yellow Soil": {
        "moisture": "Moderate",
        "crops": ["Potato", "Pulses"],
        "health_score": 58,
        "suggestion": "Use nutrient-rich fertilizers."
    }
}

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (224, 224))

    # MobileNet preprocessing
    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image

# -----------------------------
# RANDOM FOREST FERTILITY
# -----------------------------

def predict_fertility(soil_type):

    sample_data = {

        "Alluvial Soil": [138, 8.6, 560, 7.46, 0.62, 0.7, 5.9, 0.24, 0.31, 0.77, 8.71, 0.11],

        "Arid Soil": [80, 4.2, 300, 6.5, 0.4, 0.3, 3.1, 0.15, 0.20, 0.40, 4.5, 0.05],

        "Black Soil": [160, 10.0, 600, 7.8, 0.7, 0.8, 6.5, 0.30, 0.40, 0.90, 9.2, 0.15],

        "Laterite Soil": [100, 5.5, 350, 6.8, 0.5, 0.4, 4.2, 0.18, 0.25, 0.55, 5.8, 0.08],

        "Mountain Soil": [120, 6.5, 400, 6.9, 0.6, 0.5, 5.0, 0.22, 0.29, 0.65, 6.5, 0.10],

        "Red Soil": [90, 5.0, 320, 6.3, 0.45, 0.35, 3.8, 0.16, 0.21, 0.50, 5.0, 0.06],

        "Yellow Soil": [85, 4.8, 310, 6.2, 0.42, 0.33, 3.5, 0.14, 0.19, 0.45, 4.8, 0.05]
    }

    columns = [
        "N", "P", "K", "pH", "EC", "OC",
        "S", "Zn", "Fe", "Cu", "Mn", "B"
    ]

    input_data = pd.DataFrame(
        [sample_data[soil_type]],
        columns=columns
    )

    prediction = rf_model.predict(input_data)

    fertility_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return fertility_map.get(int(prediction[0]), "Medium")

# -----------------------------
# MAIN SOIL PREDICTION
# -----------------------------

def predict_soil(image_path):

    # Preprocess image
    processed_image = preprocess_image(image_path)

    # Predict probabilities
    prediction = model.predict(processed_image)[0]

    # Top 3 predictions
    top_indices = prediction.argsort()[-3:][::-1]

    top_predictions = []

    for index in top_indices:

        soil_name = class_names[index]

        confidence = float(prediction[index]) * 100

        top_predictions.append({

            "soil": soil_name,

            "confidence": round(confidence, 2)
        })

    # Main prediction
    predicted_soil = top_predictions[0]["soil"]

    main_confidence = top_predictions[0]["confidence"]

    # Soil details
    soil_details = soil_info[predicted_soil]

    # Fertility prediction
    fertility_result = predict_fertility(predicted_soil)

    return {

        "soil_type": predicted_soil,

        "confidence": main_confidence,

        "top_predictions": top_predictions,

        "fertility": fertility_result,

        "moisture": soil_details["moisture"],

        "crops": soil_details["crops"],

        "health_score": soil_details["health_score"],

        "suggestion": soil_details["suggestion"]
    }