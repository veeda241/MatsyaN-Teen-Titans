# 🐟 FishVision Agent

FishVision is an intelligent assistant that identifies fish species from images using a trained MobileNetV2 model and provides rich species information including habitat, conservation status, and fun facts. Built with TensorFlow, Gradio, and modular Python workflows.

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

```bash
git clone https://github.com/your-username/FishVision-Agent.git
cd FishVision-Agent


Set Up Environment

python -m venv tf_env
source tf_env/bin/activate  # or tf_env\Scripts\activate on Windows
pip install -r requirements.txt

Run the App
python src/demo2.py
Then open your browser to http://127.0.0.1:7860

Evaluate Model
python src/evaluate.py

project Structure

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


Model Details

Architecture: MobileNetV2 (transfer learning)
Classes: 32 fish species
Input size: 224x224
Framework: TensorFlow / Keras


Optional GPT Integration
To enable GPT-based responses, set your OpenAI API key in llm_agent.py and use model="gpt-3.5-turbo"


License
MIT License — feel free to use, modify, and share.


Acknowledgments
Built by Vyas with a passion for AI, biodiversity, and real-world impact 🌍🐠

---

Let me know if you want to add screenshots, sample predictions, or a public link for demo sharing. You’ve built something worth showing off! 🐟💡📢

