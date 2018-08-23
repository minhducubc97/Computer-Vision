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
for (digit, contour) in enumerate(listOfDigits):
    # extract the bounding box for the digit and resize it appropriately
    (x, y, width, height) = cv2.boundingRect(contour)
    roi = ref_ocra[y:y + height, x:x + width]
    roi = cv2.resize(roi, (60, 90))
    digitDict[digit] = roi


# load the input image, resize it and convert to grayscale
image = cv2.imread(args["image"])
resizedImage = imutils.resize(image, width=300)
grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original image", image)
cv2.imshow("Gray image", grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

#=== TOP-HAT TRANSFORM ===#

# initialize a rectangular (its width > its height) kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))

# initialize a square kernel
sqrKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# apply a tophat (whitehat) morphological operator to find light regions against the dark background (which should be the numbers in a typical credit card)
tophat = cv2.morphologyEx(grayImage, cv2.MORPH_TOPHAT, rectKernel)
cv2.imshow("Tophat image", tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

# compute the Scharr gradient of the tophat image, then scale the rest back into the range [0, 255]
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")