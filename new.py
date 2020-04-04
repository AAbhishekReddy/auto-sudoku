import cv2 as cv
import numpy as np
import operator

# Method dump
def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
	"""Draws circular points on an image."""
	img = in_img.copy()

	# Dynamically change to a colour image if necessary
	if len(colour) == 3:
		if len(img.shape) == 2:
			img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
		elif img.shape[2] == 1:
			img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

	for point in points:
		img = cv.circle(img, tuple(int(x) for x in point), radius, colour, -1)
	cv.imshow("Points", img)
	return img

def dist(p1, p2):
	"""Returns the scalar distance between two points"""
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))


# Image reading
img = cv.imread("new.jpeg", cv.IMREAD_GRAYSCALE)

# Blurring the image
ghost = cv.GaussianBlur(img.copy(), (11, 11), 0)
cv.imshow("Blurred", ghost)

# Adaptive thresholding
ghost = cv.adaptiveThreshold(ghost, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 2)

# Inverting the colors
ghost = cv.bitwise_not(ghost, ghost)

# Dialatting
# kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
# kernel = kernel.astype(np.unit8)
kernel = np.ones((3,3),np.uint8)
kernel[0][0] = 0
kernel[0][2] = 0
kernel[2][0] = 0
kernel[2][2] = 0
print(kernel)
ghost = cv.dilate(ghost, kernel)


# Finding the contours - Only the external contours
new_img, cont, h = cv.findContours(ghost.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Sorting in ascending order based on the areas
cont = sorted(cont, key = cv.contourArea, reverse = True)
# Select the largest area one
pol = cont[0]

# Finding the corner points
br, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in pol]), key = operator.itemgetter(1))
tl, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in pol]), key = operator.itemgetter(1))
bl, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in pol]), key = operator.itemgetter(1))
tr, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in pol]), key = operator.itemgetter(1))

cor = [pol[tl][0], pol[tr][0], pol[br][0], pol[bl][0]]

# display_points(ghost, cor)
src = np.array([cor[0], cor[1], cor[2], cor[3]], dtype='float32')

side = max([
	dist(cor[2], cor[1]),
	dist(cor[0], cor[3]),
	dist(cor[2], cor[3]),
	dist(cor[0], cor[1]),
	])

dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

# Transformation matrix
m = cv.getPerspectiveTransform(src, dst)

# Transformation on the origonal image
nn_ghost = cv.warpPerspective(img, m, (int(side), int(side)))

cv.imwrite("cropped.jpeg", nn_ghost)
cv.imshow("Cropped",nn_ghost)
# ghost = cv.cvtColor(ghost, cv.COLOR_GRAY2RGB)

# ex_ghost = cv.drawContours(ghost.copy(), ext_contours, -1, (255, 0, 0), 2)

# cv.imshow("Outer Contours", ex_ghost)


                        
cv.waitKey(0)
cv.destroyAllWindows()