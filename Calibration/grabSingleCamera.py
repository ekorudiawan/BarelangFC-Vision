import cv2
import numpy as np
import glob

REMAP_INTERPOLATION = cv2.INTER_LINEAR
# Load previously saved data
with np.load('B.npz') as X:
    mapx, mapy = [X[i] for i in ('mapx','mapy')]

IMAGE_PATH = "MARKERS/{:06d}.jpg"

#if len(sys.argv) != 1:
#    print("Syntax: {0} CALIBRATION_FILE".format(sys.argv[0]))
#    sys.exit(1)

#calibration = np.load(sys.argv[1], allow_pickle=False)

#mapx = calibration["mapx"]
#mapy = calibration["mapy"]


    #dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)


#CAMERA_WIDTH = 1300#1280
#CAMERA_HEIGHT = 720

# TODO: Use more stable identifiers
cap = cv2.VideoCapture(0)

# Increase the resolution
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
#left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
#right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
#right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

frameId = 0

# The distortion in the left and right edges prevents a good calibration, so
# discard the edges
#CROP_WIDTH =1280 #960 1280
#def cropHorizontal(image):
#    return image[:,
#            int((CAMERA_WIDTH-CROP_WIDTH)/2):
#            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]

# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    
    if not cap.grab() :
        print("No more frames")
        break

    _, Frame = cap.retrieve()
    
    fixedFrame = cv2.remap(Frame, mapx, mapy, REMAP_INTERPOLATION)
    #dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    cv2.imshow('frame', fixedFrame)
    
    if cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite(IMAGE_PATH.format(frameId), fixedFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
