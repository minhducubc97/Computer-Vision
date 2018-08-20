################################## COMMON SETUP ################################

# import the necessary packages
from imutils import contours
import numpy as np
import argparse
import cv2
import imutils

#################################### ARGUMENT ##################################

# construct the argument parse and parse the argumnents
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-r", "--reference", required=True, help="path to reference OCR-A image")
args = vars(ap.parse_args())

##################################### CONSTANTS #####################################

# define a dictionary that links the first digit of a credit card number to the credit card type
FIRST_DIGIT = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

###################################### MAIN ######################################
