import cv2
import numpy as np
import glob

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
 
	# find the contours in the edged image
	#  and keep the largest one;
	( _, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key = cv2.contourArea)
 	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)
 
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth
 
#KNOWN_DISTANCE = 60.0
#KNOWN_WIDTH = 11 #122
KNOWN_DISTANCE = 40.0
KNOWN_WIDTH = 10 #122
 
# initialize the list of images that we'll be using
IMAGE_PATHS = ["MARKERS/40c.jpg"]
image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
print marker[1][0]
#----------------------------------------------------------------

REMAP_INTERPOLATION = cv2.INTER_LINEAR
# Load previously saved data
with np.load('B.npz') as X:
    mapx, mapy = [X[i] for i in ('mapx','mapy')]

# TODO: Use more stable identifiers
cap = cv2.VideoCapture(0)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    
    if not cap.grab() :
        print("No more frames")
        break

    _, Frame = cap.retrieve()
    
    fixedFrame = cv2.remap(Frame, mapx, mapy, REMAP_INTERPOLATION)

#    radius = 50
#    focalLength = (radius * KNOWN_DISTANCE) / KNOWN_WIDTH
#    cm = distance_to_camera(KNOWN_WIDTH, focalLength, radius)

    marker = find_marker(fixedFrame)
    cm = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
 	# draw a bounding box around the image and display it
    box = np.int0(cv2.boxPoints(marker))

    cv2.drawContours(fixedFrame, [box], -1, (0, 255, 0), 2)
    cv2.putText(fixedFrame, "%.fcm" % (cm), (fixedFrame.shape[1] - 200, fixedFrame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.imshow('frame', fixedFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
