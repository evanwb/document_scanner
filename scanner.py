# import packages
import numpy as np
import cv2
import imutils
import argparse 
from skimage.filters import threshold_local
from transformations import *

# construct argument parser and parse args
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', default='images/receipt.jpg')
args = vars(ap.parse_args())

# load image, compute the ratio of old height 
# to the new height, clone and resize
image = cv2.imread(args['image'])
ratio = image.shape[0] / 500.0
original = image.copy()    
image = imutils.resize(image, height=500)

# convert image to gray, blur and apply edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

# show original image and edge detected image 
print('applying edge detection')
cv2.imshow('Original', image)
#cv2.imshow('Edged', edged)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# find contours and keep largest one
# initialize screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]


# loop through contours
for c in cnts:
    # approximate contours
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)

    # check if approximated contour has four points
    if len(approx) == 4:
        screen_contours = approx
        break
    else:
        print('Contours not detected')
        break

# show contour outline of document
print('finding contours of document')
cv2.drawContours(image, [screen_contours], -1, (255,0,0), 2)
image = imutils.resize(image, width=image.shape[0])
cv2.imshow('Outline', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# apply four point transform
warped = four_point_transform(original, screen_contours.reshape(4,2)* ratio)

# convert to grayscale then threshold it
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method='gaussian')
warped = (warped>T).astype('uint8') * 255

# show original image and scanned images
print('[info] applying perspective transform')
cv2.imshow('Original', imutils.resize(original, height=650))
cv2.imshow('Scanned', imutils.resize(warped, height=650))
name = args['image'][:len(args['image'])-4]
filename = name+'_scanned.jpg'
cv2.imwrite(filename, warped)
print('saved scan: {}'.format(filename))
cv2.waitKey(0)
