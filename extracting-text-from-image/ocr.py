################################## COMMON SETUP ################################

# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os

#################################### ARGUMENT ##################################

# construct the argument parse and parse the argumnents
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing to be done") # either thresh (threshold) or blur
args = vars(ap.parse_args())

#################################### STEP 1: PROCESS IMAGE ##################################

# read image
image = cv2.imread(args["image"])
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply preprocess => get a clear image without noise
if args["preprocess"] == "thresh":
    # segment foreground and background with THRESH_BINARY and THRESH_OTSU
    grayImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif args["preprocess"] == "blur":
    # reduce noise
    grayImage = cv2.medianBlur(grayImage, 3)

# write the grayscale image to disk as a temporary file in order to apply OCR
filename = "{}.png".format(os.getpid()) # derive a temporary image name based on the python script
cv2.imwrite(filename, grayImage)

# for Window built, it is necessary to specify the path and environment variables as follow:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
tessdata_dir_config = r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'

# load the image in PIL/Pillow form and apply OCR
text = pytesseract.image_to_string(Image.open(filename), lang='eng', config=tessdata_dir_config)

# remove the temporary file
os.remove(filename)
print(text)

# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", grayImage)
cv2.waitKey(0)

# save the output
outputImageDir = "./images/output" + args["image"][7:]
cv2.imwrite(outputImageDir, grayImage)