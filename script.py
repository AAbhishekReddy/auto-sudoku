import pytesseract as tess
from PIL import Image


img = Image.open('suduko.jpg')
text = tess.image_to_string(img)

print(text)