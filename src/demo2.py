import gradio as gr
from llm_agent import run_agent  # Still using the simplified version

def full_agent(image, question):
    print("📷 Image received as PIL object")
    print("❓ Question:", question)

    temp_path = "temp_image.jpg"
    image.save(temp_path)

    result = run_agent(question, image_path=temp_path)
    print("🧠 Result:", result)
    return result

gr.Interface(
    fn=full_agent,
    inputs=[
        gr.Image(type="pil", label="Upload Fish Image"),
        gr.Textbox(label="Ask a Question")
    ],
    outputs="text",
    title="FishVision Agent",
    description="""Upload a fish image and get species info.
    Created by S.Vyas for Fish Analysis project. from the team of Teen Titans""",
).launch()
