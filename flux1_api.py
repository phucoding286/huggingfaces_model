import requests

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

print("đang tạo ảnh ...")
image_bytes = query({
	"inputs": "a sexy anime girl",
})
print("quá trình đã xong")


import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))
image.save("output_image.png")