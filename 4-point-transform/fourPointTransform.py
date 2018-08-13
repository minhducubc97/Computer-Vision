# import the necessary packages
import numpy as np
import cv2

# input: pts, a list of four points along with their corresponding (x,y) coordinates of the target rectangle
def order_points(pts):
    # initialize a list of 4 corner points: top-left, top-right, bottom-left, bottom-right
    rect = np.zeros((4,2), dtype = "float32") # 4 rows 2 columns empty array, data type is float32

    # Sum along the axis = 1:
    # Eg: pts is a LIST contains four points: [[0,0] [10,0] [10,10] [0,10]] => sum: [0, 10, 20, 10]
    theSum = np.sum(pts, axis = 1)
    # set the top-left point to have the smallest sum, the bottom-right to have the greatest sum
    rect[0] = pts[np.argmin(theSum)]
    rect[2] = pts[np.argmax(theSum)]

    # Difference along the axis = 1:
    theDif = np.diff(pts, axis = 1)
    # set the top-right point to have the smallest difference, the bottom-left to have the greatest difference
    rect[1] = pts[np.argmin(theDif)]
    rect[3] = pts[np.argmax(theDif)]

    # return this new array of coordinates
    return rect 