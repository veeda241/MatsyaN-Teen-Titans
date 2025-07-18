# 🐟 PiscisAI

PiscisAI is your friendly fish-identifying assistant! Just snap or upload a photo of a fish, and PiscisAI will tell you what species it is — instantly. Powered by a trained MobileNetV2 model, it doesn’t just stop at identification. You’ll also get interesting info about the fish’s habitat, conservation status, and even some fun facts.

Built with TensorFlow, Gradio, and clean, modular Python code, PiscisAI is designed to be simple, fast, and surprisingly fun to use — whether you're a researcher, hobbyist, or just curious about the fish you saw on your last trip.

---

## 📦 Features

- 🧠 Deep learning-based fish species classification (32 classes)
- 📘 Species metadata integration (habitat, status, fun facts)
- 🖼️ Gradio-powered image upload and interactive Q&A
- 💬 Optional GPT integration for conversational responses
- 📊 Evaluation script with classification report

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

git clone https://github.com/veeda241/FishVision-Agent.git
cd FishVision-Agent


### ✅ Set Up Environment

python -m venv tf_env
source tf_env/bin/activate  # or tf_env\Scripts\activate on Windows
pip install -r requirements.txt

### ▶️ Run the App

python src/demo2.py
Then open your browser to http://127.0.0.1:7860

### 📊 Evaluate Model

python src/evaluate.py

### 🗂 Project Structure

FishVision-Agent/
├── fish-dataset/              # Optional: sample images
├── src/
│   ├── agent_response.py      # Model prediction + species info
│   ├── config.py              # Model path + image size
│   ├── demo2.py               # Gradio interface
│   ├── evaluate.py            # Model evaluation
│   ├── llm_agent.py           # GPT integration (optional)
│   ├── species_info.json      # Metadata for each fish
│   ├── models/
│   │   └── fish_classifier_model.keras
│   └── data_setup.py          # Data generators
├── requirements.txt
└── README.md


### 🧠 Model Details

Architecture: MobileNetV2 (transfer learning)
Classes: 32 fish species
Input size: 224x224
Framework: TensorFlow / Keras


### 🤖 Optional GPT Integration

To enable GPT-based responses, set your OpenAI API key in llm_agent.py and use model="gpt-3.5-turbo"


### 📜 License

MIT License — feel free to use, modify, and share.

### 🙌 Acknowledgments

Built by Vyas with a passion for AI, biodiversity, and real-world impact 🌍🐠
### YOUTUBE LINK:
https://youtu.be/yd_yPEaVIUI

### ppt file:
https://robinhoodai.my.canva.site/fishvision

---
