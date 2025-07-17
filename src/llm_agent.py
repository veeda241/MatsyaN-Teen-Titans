from agent_response import agent_response
from PIL import Image

def run_agent(user_message: str, image_path: str = None) -> str:
    if not image_path:
        return "❌ No image provided."

    try:
        image = Image.open(image_path)
        fish_response = agent_response(image)
        return f"🧠 FishVision Response:\n\n{fish_response}"
    except Exception as e:
        return f"❌ Failed to load image or run prediction: {str(e)}"
