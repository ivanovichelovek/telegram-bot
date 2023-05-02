import requests
from numpy import asarray
from PIL import Image

response = requests.get(file_url)
with open('img.jpg', 'wb') as f:
    f.write(response.content)
with Image.open("img.jpg") as img:
    img = asarray(img.resize((100, 100))) / 255
    print(img.shape)
