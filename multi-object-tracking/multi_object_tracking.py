# USAGE
# python multi_object_tracking.py --video videos/#filename.extension --tracker #typeoftracker

# Import the necessary packages
from imutils.video import VideoStream
import argparse # parse arguments
import imutils # a module specificly used to resize, rotate and crop images
import time
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser() 	# Specially designed for command line string
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type") # Set Kcf to be the default tracking algorithm
args = vars(ap.parse_args())	# vars() return a dictionary, which means args is a dictionary

# Initialize a dictionary that maps strings to their corresponding OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	# NOTE: These functions sometimes go missing in the official distribution, so in these cases, to solve the problem, install opencv-contrib: 
	# pip install opencv-contrib-python
	"csrt": cv2.TrackerCSRT_create,                  	# RECOMMENDED: A bit slow but very accurate
	"kcf": cv2.TrackerKCF_create,                     	# RECOMMENDED: Averagely fast and averagely accurate
	"boosting": cv2.TrackerBoosting_create,				# Slow and not very accurate (and the oldest)
	"mil": cv2.TrackerMIL_create,						# Averagely accurate, but handle errors badly
	"tld": cv2.TrackerTLD_create,						# NOT RECOMMENDED: Prone to false-positive
	"medianflow": cv2.TrackerMedianFlow_create,			# Doesn't work well when there is a sudden change in motion
	"mosse": cv2.TrackerMOSSE_create,				  	# RECOMMENDED: Extremely fast but not very accurate -> good when user needs for speed
	"goturn": cv2.TrackerGOTURN_create					# Complicated
	# These are 8 types of tracker, by default kcf is chosen
}

# Initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

# If a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):						# get(#KEY, #DEFAULT_RETURNING_VALUE)
	print("[INFO]: Starting webcam...")
	# activate webcam
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# Otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=700, height=500)

	# grab the updated bounding box coordinates (if any) for each
	# object that is being tracked
	(success, boxes) = trackers.update(frame)

	# loop over the bounding boxes and update drawing them then on the frame
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		box = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True) 

		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, box)

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()