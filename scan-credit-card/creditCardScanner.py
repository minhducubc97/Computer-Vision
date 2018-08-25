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

####################################### CONSTANTS ####################################

# define a dictionary that links the first digit of a credit card number to the credit card type
FIRST_DIGIT = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

############################ STEP 1: PREPARE THE REFERENCE ############################

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

################################ STEP 2: PREPARE THE IMAGE ################################

# load the input image, resize it and convert to grayscale
image = cv2.imread(args["image"])
resizedImage = imutils.resize(image, width=300)
grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original image", resizedImage)
cv2.imshow("Gray image", grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################ STEP 3: TOP-HAT TRANSFORM ################################

# initialize a rectangular (its width > its height) kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))

# initialize a square kernel
sqrKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# apply a tophat (whitehat) morphological operator to find light regions against the dark background (which should be the numbers in a typical credit card)
tophat = cv2.morphologyEx(grayImage, cv2.MORPH_TOPHAT, rectKernel)
cv2.imshow("Tophat image", tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################# STEP 4: APPLY SOBEL OPERATOR #############################

# compute the Scharr gradient of the tophat image, then scale the rest back into the range [0, 255]
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1) # => reduce noise and emphasize edges
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
cv2.imshow("Scharr gradient of the image", gradX)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################ STEP 5: CLOSING OPERATION ################################ 

# apply a closing operation using the above rectangular kernel to close the gaps between the credit card digits
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

# apply Otsu's thresholding operation to binarize the image
thresholdedImage = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# apply a second closing operation using the above square kernel to the binary image to further close the gaps between digits
thresholdedImage = cv2.morphologyEx(thresholdedImage, cv2.MORPH_CLOSE, sqrKernel)
cv2.imshow("Closed image", thresholdedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################# STEP 6: IDENTIFY DIGITS #################################

# find the contours in the thresholded image
listOfContours = cv2.findContours(thresholdedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
listOfContours = listOfContours[1] if imutils.is_cv3() else listOfContours[0]

# initialize a list of digit groups
listOfDigitGroups = []

# loop over the found list of contours
for (index, contour) in enumerate(listOfContours):
    # find the bounding box of the contour
    (x, y, width, height) = cv2.boundingRect(contour)
    # find the aspect ratio
    ratio = width / float(height)

    # as all the digits have the same size, the four groups of digits should have the same ratio; based on observation, the ratio should be between 2.0 and 4.0
    if (ratio > 2.0 and ratio < 4.0):
        # based on observations, the width is between 40 and 60 pixels, while the height is between 10 and 20
        if (width > 40 and width < 60) and (height > 10 and height < 20):
            listOfDigitGroups.append((x, y, width, height))

# sort the digit locations from left to right
listOfDigitGroups = sorted(listOfDigitGroups, key=lambda x:x[0])
finalOutput = []

# loop over 4 groups of digits on the credit card
for (index, (groupX, groupY, groupWidth, groupHeight)) in enumerate(listOfDigitGroups):
    # initialize the list of group digits
    groupOfOutput = []

    # get the ROI group of 4 digits from the grayscale image grayImage, then use thresholding to get the digits from the credit card
    groupOfFour = grayImage[(groupY - 5) : (groupY + groupHeight + 5), (groupX - 5) : (groupX + groupWidth + 5)]
    groupOfFour = cv2.threshold(groupOfFour, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # detect the contours of each individual digit in the group 
    digitContours = cv2.findContours(groupOfFour.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitContours = digitContours[1] if imutils.is_cv3() else digitContours[0]
    digitContours = contours.sort_contours(digitContours, method="left-to-right")[0]

    for contour in digitContours:
        # get the bounding box of each diti on the card
        (x, y, width, height) = cv2.boundingRect(contour)
        roi = groupOfFour[y : (y + height), x : (x + width)]
        roi = cv2.resize(roi, (60, 90))

        # initialize a list of template matching scores, the matching with the highest score will be the correct digit.
        scores = []

        # loop over the reference digit name and digit ROI
        for (digit, digitROI) in digitDict.items():
            # apply correlation-based template matching, take score and update the score list
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # the highest score is the one
        groupOfOutput.append(str(np.argmax(scores))) # np.argmax returns the index of the largest element
    
    # finish groupOfOutput

    # draw the digits on top and highlight the digits on the credit card
    cv2.rectangle(resizedImage, (groupX - 5, groupY - 5), (groupX + groupWidth + 5, groupY + groupHeight + 5), (0, 0, 255), 2)
    cv2.putText(resizedImage, "".join(groupOfOutput), (groupX - 2, groupY - 15), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)

    # create list of strings contains all the digits
    finalOutput.extend(groupOfOutput)

# display the output credit card information to the screen
print("Credit Card Type: {}".format(FIRST_DIGIT[finalOutput[0]]))
print("Credit Card Number: {}".format("".join(finalOutput)))

# enlarge the image for better view
enlargedImage = imutils.resize(resizedImage, width=500)
cv2.imshow("Processed image", enlargedImage)
cv2.waitKey(0)

# write the output image
outputImageDir = "./images/output" + args["image"][7:]
cv2.imwrite(outputImageDir, resizedImage)