#import packages
import cv2
import numpy as np

def order_points(points):
    # initalize a list of coordinates that will be ordered
    # first entry is in top left, second in top right, 
    # third in the bottom right, and foruth is in bottom left
    rectangle = np.zeros((4,2), dtype='float32')
    
    # top left will have the smallest sum and bottom right 
    # will have the largest sum
    sum = points.sum(axis=1)
    rectangle[0] = points[np.argmin(sum)]
    rectangle[2] = points[np.argmax(sum)]

    # calculate difference between the points, top right 
    # will have the smallest difference and the bottom left 
    # will have the largest difference
    difference = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(difference)]
    rectangle[3] = points[np.argmax(difference)]

    # return ordered coordinates
    return rectangle

def perspective_transform(image, points):
    # get a consistent order of points and unpack them seperately
    rectangle = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = rectangle

    # calculate the width of the new image which is the maximum 
    # distance between bottom right and bottom left or top right and top left
    widthA = np.sqrt(((bottom_right[0]-bottom_left[0])**2) + ((bottom_right[1]-bottom_left[1])**2))
    widthB = np.sqrt(((top_right[0]-top_left[0])**2)+((top_right[1]-top_left[1])**2))
    maxWidth = max(int(widthA),int(widthB))

    # calculate the height of the new image which will be maximum distance
    # between the top right and bottom right or top left and bottom left y-coordinates 
    heightA = np.sqrt(((top_right[0]-bottom_right[0])**2)+((top_right[1]-bottom_right[1])**2))
    heightB = np.sqrt(((top_left[0]-bottom_left[0])**2)+((top_left[1]-bottom_left[1])**2))
    maxHeight = max(int(heightA),int(heightB))

    # construct set of desination points to get top-down view of image
    destination = np.array([
        [0,0],
        [maxWidth-1, 0], 
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype='float32')

    # calculate the perspective transform maxtrix and apply
    transform_matrix = cv2.getPerspectiveTransform(rectangle, destination)
    warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))

    # return warped image
    return warped

    

