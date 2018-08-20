################################## COMMON SETUP ################################

# import the necessary packages
import numpy as np
import argparse
import cv2

#################################### ARGUMENT ##################################

# construct the argument parse and parse the argumnents
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

################################# STEP 1: FLIP IMAGE ############################

# load the image from disk
image = cv2.imread(args["image"])

# convert the image to grayscale and flip the color: the text has white color while the background is black
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayImageFlipped = cv2.bitwise_not(grayImage)

# threshold the image, setting all text pixels to max (= 255) and set all background pixels to min (0)
thresholdImage = cv2.threshold(grayImageFlipped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imshow("Original image", image)
cv2.imshow("Flipped image", thresholdImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# grab all the (x, y) coordinates of pixels that have intensities greater than 0, then use these coordinates to computed a rotated bounding box containing all the coordinates
listOfCoords = np.column_stack(np.where(thresholdImage > 0)) # column_stack combines 1-D arrays as columns into 2-D arrays

# cv2.minAreaRect() return a box in 2D structure (center (x,y), (width, height), angle of rotation); the angle is in the range [-90, 0)
angle = cv2.minAreaRect(listOfCoords)[2] 

# reverse the angle value
angle = -angle

# shift the angle appropriately to get the right angle value]
# ASSUME: the angle in our own algorithm will be counted clockwise, with respective to 0 degree lying on horizontal line; to keep track of value, the final angle will be in the range (-45,45]
if angle > 45:
    angle = -45 + (angle - 45)

# rotate the image to deskew it
(height, width) = image.shape[:2] # image.shape return height/number of rows, width/number of columns and channels 
center = (width // 2, height // 2) # in Python 3, // does integer division, while / does floating-point division
matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotatedImage = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# draw the correction angle on the image in order to validate it
cv2.putText(rotatedImage, "Angle: {:.2f} degrees".format(angle), (40, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2) #NOTE: in open cv, color scalar is BGR instead of RGB

# show the output image
print("[INFO] Angle: {:.2f}".format(angle))
cv2.imshow("Original image", image)
cv2.imshow("Deskewed image", rotatedImage)
cv2.waitKey(0)

# write the output image
outputImageDir = "./images/output" + args["image"][7:]
cv2.imwrite(outputImageDir, rotatedImage)