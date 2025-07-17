import os
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from config import IMAGE_SIZE, MODEL_PATH

# ✅ Load trained model safely
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    model = None

# ✅ Species labels (must match training classes)
class_labels = [
    "Bangus", "Big Head Carp", "Black Spotted Barb", "Catfish",
    "Climbing Perch", "Fourfinger Threadfin", "Freshwater Eel",
    "Glass Perchlet", "Goby", "Gold Fish", "Gourami", "Grass Carp",
    "Green Spotted Puffer", "Indian Carp", "Indo-Pacific Tarpon",
    "Jaguar Gapote", "Janitor Fish", "Knifefish", "Long-Snouted Pipefish",
    "Mosquito Fish", "Mudfish", "Mullet", "Pangasius", "Perch", "Scat Fish",
    "Silver Barb", "Silver Carp", "Silver Perch", "Snakehead", "Tenpounder",
    "Tilapia"
]

# ✅ Load species info safely
try:
    json_path = os.path.join(os.path.dirname(__file__), "species_info.json")
    with open(json_path, "r") as f:
        species_lookup = json.load(f)
    print("✅ Species info loaded.")
except Exception as e:
    print(f"❌ Failed to load species_info.json: {str(e)}")
    species_lookup = {}

# ✅ Prediction + conversational response
def agent_response(image: Image.Image) -> str:
    if model is None:
        return "❌ Model not loaded. Please check MODEL_PATH in config.py."

    try:
        image = image.resize(IMAGE_SIZE)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        class_index = np.argmax(predictions)
        label = class_labels[class_index]
        confidence = round(100 * np.max(predictions), 2)

        info = species_lookup.get(label, {})
        habitat = info.get("habitat", "Unknown")
        fun_fact = info.get("fun_fact", "N/A")
        status = info.get("conservation_status", "Unknown")

        return (
            f"🧠 Species: **{label}**\n"
            f"🎯 Confidence: **{confidence}%**\n"
            f"🌍 Habitat: {habitat}\n"
            f"🛡️ Conservation Status: {status}\n"
            f"📘 Fun Fact: {fun_fact}"
        )
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"
