################################## COMMON SETUP ################################

# import the necessary packages
from fourPointTransform import four_point_transform
from skimage.filters import threshold_local # get a black and white image/increase picture contrast
import numpy as np
import argparse # parse arguments
import cv2
import imutils # a module specificly used to resize, rotate and crop images
from fpdf import FPDF

#################################### ARGUMENT ##################################

# construct the argument parse and parse the argumnents
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required="True", help = "path to the input image file")
args = vars(ap.parse_args())

############################## STEP 1: Edge Detection #########################

# load the image, clone it and resize appropriately
originalImage = cv2.imread(args["image"])
ratio = originalImage.shape[0] / 500.0 # image.shape return (rows, columns, channels)
cloneImage = originalImage.copy()
cloneImage2 = originalImage.copy()
image = imutils.resize(cloneImage, height = 500)

# convert the image to grayscale in order to find contours in step 2
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# filter out weak intensity points by blurring it using GaussianBlur and then select points that are within limited intensity range
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 75, 200) # find edges that have intensity between 75 and 200

# for viewing the original image, it is necessary to resize the original image since it can be too large
originalImageResized = imutils.resize(originalImage, height = 500) 
cv2.imshow("Original image", originalImageResized)
cv2.imshow("Edged image", edged)
cv2.waitKey(0) # wait infinitely, until key event
cv2.destroyAllWindows()

########################## STEP 2: Find contours of paper #####################

# find the contours in the edged image, keeping only the one with the largest perimeter
listOfContours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # RETR_LIST: retrieve mode - in a list, CHAIN_APPROX_SIMPLE: remove all redundant points and compresses the contour

# for cv2, the largest contours are stored at index 0; for cv3, the largest contours are at index 1
largestLOC = listOfContours[0] if imutils.is_cv2() else listOfContours[1]
# sort the list according contour area from the largest to the smallest, take the first 3
largestLOC = sorted(largestLOC, key = cv2.contourArea, reverse = True)[:3]

for contour in largestLOC:
    # approximate the shape of the contour (to be a rectangle); the accuracy is chosen 2% of arc length in this case.
    perimeter = cv2.arcLength(contour, True)
    accuracy = perimeter * 0.02
    approxShape = cv2.approxPolyDP(contour, accuracy, True)
    # if the approximated contour has 4 points (4 corners) => safely say that we found our contour
    if len(approxShape) == 4:
        highlightImage = approxShape
        break

# display the image with contour highlighted
cv2.drawContours(originalImageResized, [highlightImage], -1, (0,255, 0), 2)
cv2.imshow("Highlighted contour image", originalImageResized)
cv2.waitKey(0)
cv2.destroyAllWindows()

##################### STEP 3: Apply perspective transform #####################
    
# make the array highlightImage have 4 rows and 2 columns as required, then get the warped image
warpedImage = four_point_transform(cloneImage2, highlightImage.reshape(4, 2) * ratio) 
# convert the warped image to grayscale
warpedImage = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)

# scanner effect
T = threshold_local(warpedImage, 11, method = "gaussian", offset = 10)
warpedImage = (warpedImage > T).astype("uint8") * 255

cv2.imshow("Original image", imutils.resize(cloneImage2, height = 650))
cv2.imshow("Scanned image", imutils.resize(warpedImage, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("./images/outputImage2" + ".jpg", warpedImage)

######################### STEP 4: Generate Pdf #########################
# set directory
# pdfFile = FPDF()
# pdfFile.add_page()
# pdfFile.image(str(warpedImage), 0, 0)
# pdfFile.output("/pdf/document.pdf", "F")