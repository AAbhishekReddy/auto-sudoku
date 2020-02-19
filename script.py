import numpy as np 
import cv2 as cv
import pytesseract as tess
from PIL import Image as im 

cap = cv.VideoCapture(0)

while True:
	print("here")
	ret, frame = cap.read()
	print("not here")
	frame = cv.flip(frame, 1)

	# cv.imshow('frame', frame)
	cv.imwrite('frame.png', frame)

	img = im.read('frame.png')
	text = tess.image_to_string(img)

	print(frame)

	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()