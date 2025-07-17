from PIL import Image
from agent_response import agent_response

image = Image.open("D:/vyas/Fish-anaylsis-main/fish-dataset/train/Black Spotted Barb/Black Spotted Barb 15.jpg")

result = agent_response(image)
print("✅ Result from agent_response:")
print(result)
# This code snippet demonstrates how to use the agent_response function with a sample image.