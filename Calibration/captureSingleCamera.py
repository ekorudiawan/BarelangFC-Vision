import numpy as np
import cv2


#IMAGE_PATH = "IMAGE/{:06d}.jpg"
IMAGE_PATH = "IMAGES/{:06d}.jpg"

#CAMERA_WIDTH = 1920
#CAMERA_HEIGHT = 1080

# TODO: Use more stable identifiers
cap = cv2.VideoCapture(0)

# Increase the resolution
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# The distortion in the left and right edges prevents a good calibration, so
# discard the edges

#CROP_WIDTH = 960
#def cropHorizontal(image):
#    return image[:,
#            int((CAMERA_WIDTH-CROP_WIDTH)/2):
#            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]

frameId = 0

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    if not(cap.grab()):
       print("No more frames")
       break

    _, Frame = cap.retrieve()
    #Frame = cropHorizontal(Frame)
    

    if cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite(IMAGE_PATH.format(frameId), Frame)

    cv2.imshow('frame', Frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frameId += 1

cap.release()
cv2.destroyAllWindows()
