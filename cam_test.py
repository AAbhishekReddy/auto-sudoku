import numpy as np 
import cv2 as cv

cap = cv.VideoCapture(0)

while True:
	
	ret, frame = cap.read()

	# frame = abs(255 - frame)
	# frame = cv.bitwise_not(frame)
	# frame = ~frame
	frame = cv.flip(frame, 1)

	cv.imshow('frame', frame)

	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()