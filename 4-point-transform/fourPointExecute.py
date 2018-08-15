# import the necessary packages
# import the helper function from the file
from fourPointTransform import four_point_transform
import numpy as numpy
import argparse
import cv2

# construct the argument parse and parse the argumnents
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the input image file")
ap.add_argument("-c", "--coords", help = "comma seperated list of source points")
args = vars(ap.parse_args())

# load the input image and get the list of source points
image = cv2.imread(args["image"]) 
pts = np.array(eval(args["coords"]), dtype = "float32")

warped_image = four_point_transform(image, pts)

cv2.imshow("Original picture", image)
cv2.imshow("Front view of the picture", warped_image)
cv2.waitKey(0)