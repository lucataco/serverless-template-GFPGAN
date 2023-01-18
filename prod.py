import sys
import base64
from io import BytesIO
from PIL import Image
import banana_dev as banana

api_key = "YOUR_API_KEY_HERE"
model_key = "YOUR_MODEL_KEY"

#Read filename to read as base64 encoding
img_name = sys.argv[1:][0]
with open(img_name, "rb") as f:
    bytes = f.read()
    encoded = base64.b64encode(bytes).decode('utf-8')

model_inputs = {'img_bytes': encoded }
out = banana.run(api_key, model_key, model_inputs)

image_byte_string = out.json()["image_base64"]
image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")