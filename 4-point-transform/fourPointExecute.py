# import the necessary packages
# import the helper function from the file
from fourPointTransform import four_point_transform
import numpy as np
import argparse # parse arguments
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
# cv2.waitKey(#number) will wait for the specified #number. However, cv2.waitKey(0) will display the window infinitely till keypress event.
cv2.waitKey(0)

# Command to run the program on the images: 
# python transform_example.py --image images/example_01.png --coords "[(73, 239), (356, 117), (475, 265), (187, 443)]"
# python transform_example.py --image images/example_02.png --coords "[(101, 185), (393, 151), (479, 323), (187, 441)]"
# python transform_example.py --image images/example_03.png --coords "[(63, 242), (291, 110), (361, 252), (78, 386)]"