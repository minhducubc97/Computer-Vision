# import the necessary packages
import numpy as np
import cv2

# INPUT: pts, a list of four points along with their corresponding (x,y) coordinates of the target rectangle
# OUTPUT: a 2D array
def order_points(pts):
    # initialize a list of 4 corner points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4,2), dtype = "float32") # Empty array that has 4 rows, each row has a pair of coordinates, data type is float32

    # Sum along the axis = 1:
    # Eg: pts is a LIST contains four points: [[0,0] [10,0] [10,10] [0,10]] => sum along the axis = 1: [0, 10, 20, 10]
    theSum = np.sum(pts, axis = 1)
    # set the top-left point to have the smallest sum, the bottom-right to have the greatest sum
    rect[0] = pts[np.argmin(theSum)] # first row (top-left)
    rect[2] = pts[np.argmax(theSum)] # third row (bottom-right)

    # Difference along the axis = 1:
    # Eg: pts is a LIST contains four points: [[0,0] [10,0] [10,10] [0,10]] => difference along the axis = 1: [0, 10, 0, -10]
    theDif = np.diff(pts, axis = 1)
    # set the top-right point to have the smallest difference, the bottom-left to have the greatest difference
    rect[1] = pts[np.argmin(theDif)] # second row (top-right)
    rect[3] = pts[np.argmax(theDif)] # fourth row (bottom-left)

    # return this new array of coordinates
    return rect 

def four_point_transform(image, pts):
    # get a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (topLeftPoint, topRightPoint, bottomRightPoint, bottomLeftPoint) = rect

    # calculate the width of the new image, which will be the maximum distance between top-right and top-left x-coordinates or bottom-right and bottom-left x-coordinates
    topWidth = np.sqrt(((topRightPoint[0] - topLeftPoint[0]) ** 2) + ((topLeftPoint[1] - topRightPoint[1]) ** 2))
    bottomWidth = np.sqrt(((bottomRightPoint[0] - bottomLeftPoint[0]) ** 2) + ((bottomLeftPoint[1] - bottomRightPoint[1]) ** 2))
    
    # choose the max width between the 2 calculated above to go with
    maxWidth = max(int(topWidth), int(bottomWidth))

    # calculate the height of the new image, which will be the maximum distance between top-right and bottom-right y-coordinates or top-left and bottom-left y-coordinates
    leftHeight = np.sqrt(((topLeftPoint[0] - bottomLeftPoint[0]) ** 2) + ((topLeftPoint[1] - bottomLeftPoint[1]) ** 2))
    rightHeight = np.sqrt(((topRightPoint[0] - bottomRightPoint[0]) ** 2) + ((topRightPoint[1] - bottomRightPoint[1]) ** 2))

    # choose the max height between the 2 calculated above to go with
    maxHeight = max(int(leftHeight), int(rightHeight))

    # create the destination points (order: top-left, top-right, bottom-right and bottom-left) to obtain a front view of the image
    dst = np.array(
        [[0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype = "float32")

    # compute the actual perspective transform matrix and apply it
    PTMatrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, PTMatrix, (maxWidth, maxHeight))

    # return the warped image
    return warped