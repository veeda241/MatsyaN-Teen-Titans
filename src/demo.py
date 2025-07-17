import gradio as gr
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from config import IMAGE_SIZE, MODEL_PATH

# 🔍 Load trained model
model = load_model(MODEL_PATH)

# 🐠 Species labels (must match training folders)
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

# 📘 Load species info
try:
    with open("species_info.json", "r") as f:
        species_lookup = json.load(f)
except Exception as e:
    print("❌ Error loading species_info.json:", e)
    species_lookup = {}

# 🔮 Prediction function
def predict_fish(image):
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

    return label, confidence, habitat, fun_fact, status

# 🌐 Gradio Interface with Card Layout
with gr.Blocks(title="Fish Species Recognizer") as demo:
    gr.Markdown("## 🐟 Fish Species Recognizer")
    gr.Markdown("Upload a fish image to identify its species and learn more about it.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Fish Image")
            predict_btn = gr.Button("🔍 Analyze")

        with gr.Column():
            species_output = gr.Text(label="🧠 Predicted Species")
            confidence_output = gr.Text(label="🎯 Confidence")
            habitat_output = gr.Text(label="🌍 Habitat")
            fact_output = gr.Text(label="📘 Fun Fact")
            status_output = gr.Text(label="🛡️ Conservation Status")

    predict_btn.click(
        fn=predict_fish,
        inputs=image_input,
        outputs=[
            species_output,
            confidence_output,
            habitat_output,
            fact_output,
            status_output
        ]
    )

if __name__ == "__main__":
    demo.launch()
