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

# load the reference to OCR-A image from work directory, convert it to grayscale and threshold it to achieve white on black effect
ref_ocra = cv2.imread(args["reference"])
ref_ocra = cv2.cvtColor(ref_ocra, cv2.COLOR_BGR2GRAY)
ref_ocra = cv2.threshold(ref_ocra, 10, 255, cv2.THRESH_BINARY_INV)[1]

# extract contours from the OCR-A image, sort them from left to right 
listOfDigits = cv2.findContours(ref_ocra.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
listOfDigits = listOfDigits[1] if imutils.is_cv3() else listOfDigits[0]
listOfDigits = contours.sort_contours(listOfDigits, method="left-to-right")[0]

# create a dictionary that maps the digit name to its appropriate ROI
digitDict = {}