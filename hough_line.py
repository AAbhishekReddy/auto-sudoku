import cv2 as cv
import numpy as np

img = cv.imread('suduko.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('test', img)