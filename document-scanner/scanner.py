# import the necessary packages
from fourPointTransform import four_point_transform
from skimage.filters import threshold_local # get a black and white image/increase picture contrast
import numpy as np
import argparse # parse arguments
import cv2
import imutils # a module specificly used to resize, rotate and crop images

# construct the argument parse and parse the argumnents
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required="True", help = "path to the input image file")
args = vars(ap.parse_args())

# load the image, clone it and resize appropriately
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0 # image.shape return (rows, columns, channels)
cloneImage = image.copy()
image = imutils.resize(cloneImage, height = 500)