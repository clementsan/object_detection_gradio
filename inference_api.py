from gradio_client import Client, handle_file
from PIL import Image
import matplotlib.pyplot as plt

# Parameters
Server_URL = "cvachet/object_detection_gradio"
Image_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'

# URL to client server (e.g. local server, docker container or AWS ECS)
client = Client(Server_URL)

# Call to API
result = client.predict(
		image=handle_file(Image_URL),
		model_id="hustvl/yolos-small",
		threshold=0.9,
		api_name="/detect"
)

# Result is an image.webp file
print(result)

# Display image via matplotlib
img = Image.open(result).convert("RGB")

plt.figure(figsize=(8, 5))
plt.imshow(img)
plt.axis('off')
plt.show()
