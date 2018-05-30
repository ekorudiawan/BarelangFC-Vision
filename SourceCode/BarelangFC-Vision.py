######################################################################
# BarelangFC Vision V2.0                                             #
# By : Eko Rudiawan                                                  #
#                                                                    #
# We use machine learning to solve ball and goal recognition problem #
#                                                                    #
######################################################################

# Standard imports
import os
import sys
import cv2
import numpy as np
import datetime
import socket
import time
import math
# import glob
import matplotlib.pyplot as plt
import matplotlib
import webbrowser
from flask import Flask, render_template, Response, request
from scipy.spatial import distance as dist			
from scipy.spatial import distance				
from sklearn import tree
from sklearn.externals import joblib
# from imutils import perspective
# from imutils import contours

robotID = 1
# Transform scanning coordinat to camera coordinat
# Definisi umum
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
HALF_IMAGE_WIDTH = IMAGE_WIDTH / 2
IMAGE_AREA = IMAGE_HEIGHT * IMAGE_WIDTH
# Default filename to save all data
IMAGE_PATH = "../gambar_normal_{:01d}.jpg"
#imageDatasetPath = '../Learning_Image/my_photo-' #'D:/RoboCupDataset/normal/gambar_normal_'
settingValueFilename = 'BarelangFC-SettingValue.csv'
ballDatasetFilename = 'BarelangFC-BallDataset.csv'
ballMLFilename = 'BarelangFC-BallMLModel.sav'
goalDatasetFilename = 'BarelangFC-GoalDataset.csv'
goalMLFilename = 'BarelangFC-GoalMLModel.sav'

# Global variable for thresholding
debugmode = 0
last_debug = 0
lowerFieldGr = np.zeros(3, dtype=int)
upperFieldGr = 255 * np.ones(3, dtype=int)
edFieldGr = np.zeros(2, dtype=int)
lowerBallGr = np.zeros(3, dtype=int)
upperBallGr = 255 * np.ones(3, dtype=int)
edBallGr = np.zeros(2, dtype=int)
lowerBallWh = np.zeros(3, dtype=int)
upperBallWh = 255 * np.ones(3, dtype=int)
edBallWh = np.zeros(2, dtype=int)
lowerGoalWh = np.zeros(3, dtype=int)
upperGoalWh = 255 * np.ones(3, dtype=int)
edGoalWh = np.zeros(2, dtype=int)
debugBallGr = np.zeros(10, dtype=int)
debugBallWh = np.zeros(10, dtype=int)
debugGoalWh = np.zeros(4, dtype=int)
cameraSetting = np.zeros(10, dtype=int)

# Global variable detection result
detectedBall = np.zeros(5, dtype=int)
detectedGoal = np.zeros(7, dtype=int)

host = 'localhost'
port = 2000

app = Flask(__name__)

@app.route('/')
def index():
	"""Video streaming home page."""
	templateData = {
		'robotid' : str(robotID),
	}
	return render_template('index.html', **templateData)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def showHelp():
	print '\n------------------BarelangFC-Vision-------------------------'
	print '### All Running Mode ### -----------------------------------'
	print 'Parse Field -------------------------------------------- [F]'
	print 'Parse Ball Green (Mode 1) ------------------------------ [N]'
	print 'Parse Ball White (Mode 2) ------------------------------ [N]'
	print 'Parse Goal --------------------------------------------- [G]'
	print 'Save Filter Config ------------------------------------- [S]'
	print 'Load Filter Config ------------------------------------- [L]'
	print 'Destroy All Windows ------------------------------------ [R]'
	print 'Help --------------------------------------------------- [Z]'
	print 'Exit BarelangFC-Vision --------------------------------- [X]'
	print '### Testing / Training Mode ### ----------------------------'
	print 'Next Image --------------------------------------------- [U]'
	print 'Previous Image ----------------------------------------- [J]'
	print 'Next Contour ------------------------------------------- [K]'
	print 'Previous Contour --------------------------------------- [H]'
	print '### Training Mode ### --------------------------------------'
	print 'Mark as Object ----------------------------------------- [1]'
	print 'Mark as Not Object ------------------------------------- [0]'
	print 'Train Ball Dataset and Save ML Model ------------------- [T]'
	print 'Save Ball Dataset to CSV ------------------------------- [D]'
	print 'Load Goal Dataset from CSV, Train and Save ML Model ---- [M]\n'

def createTrackbars(mode):
	cv2.namedWindow('Control')
	# Special for setting blur image
	# Only available for field setting
	# All setting parameter

	#if mode == 1: #field
	if mode == 2: #ball mode 1
		cv2.createTrackbar('Debug Ball Mode1','Control',0,1,nothing)
	elif mode == 3: #ball mode 2
		cv2.createTrackbar('Debug Ball Mode2','Control',0,1,nothing)
	elif mode == 4: #goal
		cv2.createTrackbar('Debug Goal','Control',0,1,nothing)
	elif mode == 5: #ball mode 1
		#Ax100 10000, RAx1 100, ARx100 100, WRx10 20, PWx1 100, Rx1 320, Hx1 480, wx1 640.
		cv2.namedWindow('Control2')
		cv2.createTrackbar('Ball','Control2',0,20,nothing)
		#cv2.createTrackbar('Height Max','Control2',0,480,nothing)
		#cv2.createTrackbar('Height Min','Control2',0,480,nothing)
		#cv2.createTrackbar('Width Max','Control2',0,640,nothing)
		#cv2.createTrackbar('Width Min','Control2',0,640,nothing)
		cv2.createTrackbar('Radius Max','Control2',0,320,nothing)
		cv2.createTrackbar('Radius Min','Control2',0,320,nothing)
		cv2.createTrackbar('Area Max','Control2',0,3000,nothing) #10000
		cv2.createTrackbar('Area Min','Control2',0,3000,nothing) #10000
		#cv2.createTrackbar('Rect Area Max','Control2',0,100,nothing)
		#cv2.createTrackbar('Rect Area Min','Control2',0,100,nothing)
		cv2.createTrackbar('Area Ratio Max','Control2',0,100,nothing)
		cv2.createTrackbar('Area Ratio Min','Control2',0,100,nothing)
		cv2.createTrackbar('Wh Ratio Max','Control2',0,100,nothing)
		cv2.createTrackbar('Wh Ratio Min','Control2',0,100,nothing)
		cv2.createTrackbar('Percent White Max','Control2',0,100,nothing)
		cv2.createTrackbar('Percent White Min','Control2',0,100,nothing)
	elif mode == 6: #ball mode 2
		#Ax100 10000, RAx1 100, ARx100 100, WRx10 20, PWx1 100, Rx1 320, Hx1 480, wx1 640.
		cv2.namedWindow('Control2')
		cv2.createTrackbar('Ball','Control2',0,20,nothing)
		#cv2.createTrackbar('Height Max','Control2',0,480,nothing)
		#cv2.createTrackbar('Height Min','Control2',0,480,nothing)
		#cv2.createTrackbar('Width Max','Control2',0,640,nothing)
		#cv2.createTrackbar('Width Min','Control2',0,640,nothing)
		cv2.createTrackbar('Radius Max','Control2',0,320,nothing)
		cv2.createTrackbar('Radius Min','Control2',0,320,nothing)
		cv2.createTrackbar('Area Max','Control2',0,3000,nothing) #10000
		cv2.createTrackbar('Area Min','Control2',0,3000,nothing) #10000
		#cv2.createTrackbar('Rect Area Max','Control2',0,100,nothing)
		#cv2.createTrackbar('Rect Area Min','Control2',0,100,nothing)
		cv2.createTrackbar('Area Ratio Max','Control2',0,100,nothing)
		cv2.createTrackbar('Area Ratio Min','Control2',0,100,nothing)
		cv2.createTrackbar('Wh Ratio Max','Control2',0,100,nothing)
		cv2.createTrackbar('Wh Ratio Min','Control2',0,100,nothing)
		cv2.createTrackbar('Percent White Max','Control2',0,100,nothing)
		cv2.createTrackbar('Percent White Min','Control2',0,100,nothing)
	elif mode == 7: #goal
		#Ax100 10000, RAx1 100, ARx100 100, WRx10 20, PWx1 100, Rx1 320, Hx1 480, wx1 640.
		cv2.namedWindow('Control2')
		cv2.createTrackbar('Goal','Control2',0,20,nothing)
		#cv2.createTrackbar('Height Max','Control2',0,480,nothing)
		#cv2.createTrackbar('Height Min','Control2',0,480,nothing)
		#cv2.createTrackbar('Width Max','Control2',0,640,nothing)
		#cv2.createTrackbar('Width Min','Control2',0,640,nothing)
		#cv2.createTrackbar('Area Max','Control2',0,3000,nothing) #10000
		#cv2.createTrackbar('Area Min','Control2',0,3000,nothing) #10000
		cv2.createTrackbar('Rect Area Max','Control2',0,100,nothing)
		cv2.createTrackbar('Rect Area Min','Control2',0,100,nothing)
		cv2.createTrackbar('Area Ratio Max','Control2',0,100,nothing)
		cv2.createTrackbar('Area Ratio Min','Control2',0,100,nothing)
		#cv2.createTrackbar('Wh Ratio Max','Control2',0,100,nothing)
		#cv2.createTrackbar('Wh Ratio Min','Control2',0,100,nothing)
	elif mode == 8: #setCameraParameter
		cv2.createTrackbar('brightness','Control',128,255,nothing)
		cv2.createTrackbar('contrast','Control',128,255,nothing)
		cv2.createTrackbar('saturation','Control',128,255,nothing)
		cv2.createTrackbar('white_balance_temperature_auto','Control',1,1,nothing)
		cv2.createTrackbar('white_balance_temperature','Control',4000,7500,nothing)
		cv2.createTrackbar('sharpness','Control',128,255,nothing)
		cv2.createTrackbar('exposure_auto','Control',3,3,nothing)
		cv2.createTrackbar('exposure_absolute','Control',250,2047,nothing)
		cv2.createTrackbar('exposure_auto_priority','Control',0,1,nothing)
		cv2.createTrackbar('focus_auto','Control',1,1,nothing)

	if mode >=1 and mode <= 4: #field, ball, goal
		cv2.createTrackbar('HMax','Control',255,255,nothing)
		cv2.createTrackbar('HMin','Control',0,255,nothing)
		cv2.createTrackbar('SMax','Control',255,255,nothing)
		cv2.createTrackbar('SMin','Control',0,255,nothing)
		cv2.createTrackbar('VMax','Control',255,255,nothing)
		cv2.createTrackbar('VMin','Control',0,255,nothing)
		cv2.createTrackbar('Erode','Control',0,10,nothing)
		cv2.createTrackbar('Dilate','Control',0,100,nothing)

def loadTrackbars(mode):
	# Show Field
	if mode == 1:
		cv2.setTrackbarPos('HMin', 'Control', lowerFieldGr[0])
		cv2.setTrackbarPos('SMin', 'Control', lowerFieldGr[1])
		cv2.setTrackbarPos('VMin', 'Control', lowerFieldGr[2])
		cv2.setTrackbarPos('HMax', 'Control', upperFieldGr[0])
		cv2.setTrackbarPos('SMax', 'Control', upperFieldGr[1])
		cv2.setTrackbarPos('VMax', 'Control', upperFieldGr[2])
		cv2.setTrackbarPos('Erode', 'Control', edFieldGr[0])
		cv2.setTrackbarPos('Dilate', 'Control', edFieldGr[1])
	# Show Ball Green (mode 1)
	elif mode == 2:
		cv2.setTrackbarPos('HMin', 'Control', lowerBallGr[0])
		cv2.setTrackbarPos('SMin', 'Control', lowerBallGr[1])
		cv2.setTrackbarPos('VMin', 'Control', lowerBallGr[2])
		cv2.setTrackbarPos('HMax', 'Control', upperBallGr[0])
		cv2.setTrackbarPos('SMax', 'Control', upperBallGr[1])
		cv2.setTrackbarPos('VMax', 'Control', upperBallGr[2])
		cv2.setTrackbarPos('Erode', 'Control', edBallGr[0])
		cv2.setTrackbarPos('Dilate', 'Control', edBallGr[1])
	# Show Ball White (mode 2)
	elif mode == 3:
		cv2.setTrackbarPos('HMin', 'Control', lowerBallWh[0])
		cv2.setTrackbarPos('SMin', 'Control', lowerBallWh[1])
		cv2.setTrackbarPos('VMin', 'Control', lowerBallWh[2])
		cv2.setTrackbarPos('HMax', 'Control', upperBallWh[0])
		cv2.setTrackbarPos('SMax', 'Control', upperBallWh[1])
		cv2.setTrackbarPos('VMax', 'Control', upperBallWh[2])
		cv2.setTrackbarPos('Erode', 'Control', edBallWh[0])
		cv2.setTrackbarPos('Dilate', 'Control', edBallWh[1])
	# Show Goal
	elif mode == 4:
		cv2.setTrackbarPos('HMin', 'Control', lowerGoalWh[0])
		cv2.setTrackbarPos('SMin', 'Control', lowerGoalWh[1])
		cv2.setTrackbarPos('VMin', 'Control', lowerGoalWh[2])
		cv2.setTrackbarPos('HMax', 'Control', upperGoalWh[0])
		cv2.setTrackbarPos('SMax', 'Control', upperGoalWh[1])
		cv2.setTrackbarPos('VMax', 'Control', upperGoalWh[2])
		cv2.setTrackbarPos('Erode', 'Control', edGoalWh[0])
		cv2.setTrackbarPos('Dilate', 'Control', edGoalWh[1])
	elif mode == 5:
		cv2.setTrackbarPos('Radius Max', 'Control2', debugBallGr[0])
		cv2.setTrackbarPos('Radius Min', 'Control2', debugBallGr[1])
		cv2.setTrackbarPos('Area Max', 'Control2', debugBallGr[2])
		cv2.setTrackbarPos('Area Min', 'Control2', debugBallGr[3])
		cv2.setTrackbarPos('Area Ratio Max', 'Control2', debugBallGr[4])
		cv2.setTrackbarPos('Area Ratio Min', 'Control2', debugBallGr[5])
		cv2.setTrackbarPos('Wh Ratio Max', 'Control2', debugBallGr[6])
		cv2.setTrackbarPos('Wh Ratio Min', 'Control2', debugBallGr[7])
		cv2.setTrackbarPos('Percent White Max', 'Control2', debugBallGr[8])
		cv2.setTrackbarPos('Percent White Min', 'Control2', debugBallGr[9])
	elif mode == 6:
		cv2.setTrackbarPos('Radius Max', 'Control2', debugBallWh[0])
		cv2.setTrackbarPos('Radius Min', 'Control2', debugBallWh[1])
		cv2.setTrackbarPos('Area Max', 'Control2', debugBallWh[2])
		cv2.setTrackbarPos('Area Min', 'Control2', debugBallWh[3])
		cv2.setTrackbarPos('Area Ratio Max', 'Control2', debugBallWh[4])
		cv2.setTrackbarPos('Area Ratio Min', 'Control2', debugBallWh[5])
		cv2.setTrackbarPos('Wh Ratio Max', 'Control2', debugBallWh[6])
		cv2.setTrackbarPos('Wh Ratio Min', 'Control2', debugBallWh[7])
		cv2.setTrackbarPos('Percent White Max', 'Control2', debugBallWh[8])
		cv2.setTrackbarPos('Percent White Min', 'Control2', debugBallWh[9])
	elif mode == 7:
		cv2.setTrackbarPos('Rect Area Max', 'Control2', debugGoalWh[0])
		cv2.setTrackbarPos('Rect Area Min', 'Control2', debugGoalWh[1])
		cv2.setTrackbarPos('Area Ratio Max', 'Control2', debugGoalWh[2])
		cv2.setTrackbarPos('Area Ratio Min', 'Control2', debugGoalWh[3])
	elif mode == 8:
		cv2.setTrackbarPos('brightness', 'Control', cameraSetting[0])
		cv2.setTrackbarPos('contrast', 'Control', cameraSetting[1])
		cv2.setTrackbarPos('saturation', 'Control', cameraSetting[2])
		cv2.setTrackbarPos('white_balance_temperature_auto', 'Control', cameraSetting[3])
		cv2.setTrackbarPos('white_balance_temperature', 'Control', cameraSetting[4])
		cv2.setTrackbarPos('sharpness', 'Control', cameraSetting[5])
		cv2.setTrackbarPos('exposure_auto', 'Control', cameraSetting[6])
		cv2.setTrackbarPos('exposure_absolute', 'Control', cameraSetting[7])
		cv2.setTrackbarPos('exposure_auto_priority', 'Control', cameraSetting[8])
		cv2.setTrackbarPos('focus_auto', 'Control', cameraSetting[9])

def saveConfig():
	npSettingValue = np.zeros(66, dtype=int)
	npSettingValue[0] = lowerFieldGr[0] 
	npSettingValue[1] = lowerFieldGr[1] 
	npSettingValue[2] = lowerFieldGr[2] 
	npSettingValue[3] = upperFieldGr[0] 
	npSettingValue[4] = upperFieldGr[1] 
	npSettingValue[5] = upperFieldGr[2] 
	npSettingValue[6] = edFieldGr[0] 
	npSettingValue[7] = edFieldGr[1] 

	npSettingValue[8] = lowerBallGr[0] 
	npSettingValue[9] = lowerBallGr[1] 
	npSettingValue[10] = lowerBallGr[2] 
	npSettingValue[11] = upperBallGr[0] 
	npSettingValue[12] = upperBallGr[1] 
	npSettingValue[13] = upperBallGr[2] 
	npSettingValue[14] = edBallGr[0] 
	npSettingValue[15] = edBallGr[1] 

	npSettingValue[16] = lowerBallWh[0] 
	npSettingValue[17] = lowerBallWh[1] 
	npSettingValue[18] = lowerBallWh[2] 
	npSettingValue[19] = upperBallWh[0] 
	npSettingValue[20] = upperBallWh[1] 
	npSettingValue[21] = upperBallWh[2] 
	npSettingValue[22] = edBallWh[0] 
	npSettingValue[23] = edBallWh[1] 

	npSettingValue[24] = lowerGoalWh[0] 
	npSettingValue[25] = lowerGoalWh[1] 
	npSettingValue[26] = lowerGoalWh[2] 
	npSettingValue[27] = upperGoalWh[0] 
	npSettingValue[28] = upperGoalWh[1] 
	npSettingValue[29] = upperGoalWh[2] 
	npSettingValue[30] = edGoalWh[0] 
	npSettingValue[31] = edGoalWh[1] 

	npSettingValue[32] = debugBallGr[0] 
	npSettingValue[33] = debugBallGr[1] 
	npSettingValue[34] = debugBallGr[2] 
	npSettingValue[35] = debugBallGr[3] 
	npSettingValue[36] = debugBallGr[4] 
	npSettingValue[37] = debugBallGr[5] 
	npSettingValue[38] = debugBallGr[6] 
	npSettingValue[39] = debugBallGr[7] 
	npSettingValue[40] = debugBallGr[8] 
	npSettingValue[41] = debugBallGr[9] 

	npSettingValue[42] = debugBallWh[0] 
	npSettingValue[43] = debugBallWh[1] 
	npSettingValue[44] = debugBallWh[2] 
	npSettingValue[45] = debugBallWh[3] 
	npSettingValue[46] = debugBallWh[4] 
	npSettingValue[47] = debugBallWh[5] 
	npSettingValue[48] = debugBallWh[6] 
	npSettingValue[49] = debugBallWh[7] 
	npSettingValue[50] = debugBallWh[8] 
	npSettingValue[51] = debugBallWh[9] 

	npSettingValue[52] = debugGoalWh[0] 
	npSettingValue[53] = debugGoalWh[1] 
	npSettingValue[54] = debugGoalWh[2] 
	npSettingValue[55] = debugGoalWh[3] 

	npSettingValue[56] = cameraSetting[0] 
	npSettingValue[57] = cameraSetting[1] 
	npSettingValue[58] = cameraSetting[2] 
	npSettingValue[59] = cameraSetting[3] 
	npSettingValue[60] = cameraSetting[4] 
	npSettingValue[61] = cameraSetting[5] 
	npSettingValue[62] = cameraSetting[6] 
	npSettingValue[63] = cameraSetting[7] 
	npSettingValue[64] = cameraSetting[8] 
	npSettingValue[65] = cameraSetting[9] 

	npSettingValue = np.reshape(npSettingValue, (1, 66))
	headerLabel = '''F HMin, F SMin, F SMin, F HMax, F SMax, F SMax, F Erode, F Dilate, B Gr HMin, B Gr SMin, B Gr SMin, B Gr HMax, B Gr SMax, B Gr SMax, B Gr Erode, B Gr Dilate, B Wh HMin, B Wh SMin, B Wh SMin, B Wh HMax, B Wh SMax, B Wh SMax, B Wh Erode, B Wh Dilate, G HMin, G SMin, G SMin, G HMax, G SMax, G SMax, G Erode, G Dilate'''
	np.savetxt(settingValueFilename, npSettingValue, fmt = '%d', delimiter = ',', header = headerLabel)
	print 'Setting Parameter Saved'

def loadConfig():
	csvSettingValue = np.genfromtxt(settingValueFilename, dtype=int, delimiter=',', skip_header=True)
	print csvSettingValue
	lowerFieldGr[0] = csvSettingValue[0]
	lowerFieldGr[1] = csvSettingValue[1]
	lowerFieldGr[2] = csvSettingValue[2]
	upperFieldGr[0] = csvSettingValue[3]
	upperFieldGr[1] = csvSettingValue[4]
	upperFieldGr[2] = csvSettingValue[5]
	edFieldGr[0] = csvSettingValue[6]
	edFieldGr[1] = csvSettingValue[7]

	lowerBallGr[0] = csvSettingValue[8]
	lowerBallGr[1] = csvSettingValue[9]
	lowerBallGr[2] = csvSettingValue[10]
	upperBallGr[0] = csvSettingValue[11]
	upperBallGr[1] = csvSettingValue[12]
	upperBallGr[2] = csvSettingValue[13]
	edBallGr[0] = csvSettingValue[14]
	edBallGr[1] = csvSettingValue[15]

	lowerBallWh[0] = csvSettingValue[16]
	lowerBallWh[1] = csvSettingValue[17]
	lowerBallWh[2] = csvSettingValue[18]
	upperBallWh[0] = csvSettingValue[19]
	upperBallWh[1] = csvSettingValue[20]
	upperBallWh[2] = csvSettingValue[21]
	edBallWh[0] = csvSettingValue[22]
	edBallWh[1] = csvSettingValue[23]

	lowerGoalWh[0] = csvSettingValue[24]
	lowerGoalWh[1] = csvSettingValue[25]
	lowerGoalWh[2] = csvSettingValue[26]
	upperGoalWh[0] = csvSettingValue[27]
	upperGoalWh[1] = csvSettingValue[28]
	upperGoalWh[2] = csvSettingValue[29]
	edGoalWh[0] = csvSettingValue[30]
	edGoalWh[1] = csvSettingValue[31]

	debugBallGr[0] = csvSettingValue[32]
	debugBallGr[1] = csvSettingValue[33]
	debugBallGr[2] = csvSettingValue[34]
	debugBallGr[3] = csvSettingValue[35]
	debugBallGr[4] = csvSettingValue[36]
	debugBallGr[5] = csvSettingValue[37]
	debugBallGr[6] = csvSettingValue[38]
	debugBallGr[7] = csvSettingValue[39]
	debugBallGr[8] = csvSettingValue[40]
	debugBallGr[9] = csvSettingValue[41]

	debugBallWh[0] = csvSettingValue[42]
	debugBallWh[1] = csvSettingValue[43]
	debugBallWh[2] = csvSettingValue[44]
	debugBallWh[3] = csvSettingValue[45]
	debugBallWh[4] = csvSettingValue[46]
	debugBallWh[5] = csvSettingValue[47]
	debugBallWh[6] = csvSettingValue[48]
	debugBallWh[7] = csvSettingValue[49]
	debugBallWh[8] = csvSettingValue[50]
	debugBallWh[9] = csvSettingValue[51]

	debugGoalWh[0] = csvSettingValue[52]
	debugGoalWh[1] = csvSettingValue[53]
	debugGoalWh[2] = csvSettingValue[54]
	debugGoalWh[3] = csvSettingValue[55]

	cameraSetting[0] = csvSettingValue[56] #128	# Brightness
	cameraSetting[1] = csvSettingValue[57] #128	# Contrast
	cameraSetting[2] = csvSettingValue[58] #128	# Saturation
	cameraSetting[3] = csvSettingValue[59] #0	# White_balance_temperature_auto
	cameraSetting[4] = csvSettingValue[60] #5000	# White_balance_temperature
	cameraSetting[5] = csvSettingValue[61] #128	# Sharpness
	cameraSetting[6] = csvSettingValue[62] #1	# Exposure_auto
	cameraSetting[7] = csvSettingValue[63] #312	# Exposure_absolute
	cameraSetting[8] = csvSettingValue[64] #0	# Exposure_auto_priority
	cameraSetting[9] = csvSettingValue[65] #0	# Focus_auto
	print 'Setting Parameter Loaded'

def setCameraParameter():
	print "Before Set Camera Setting Parameter"
	os.system("v4l2-ctl --list-ctrls")

	Brightness = "v4l2-ctl --set-ctrl brightness={}".format(cameraSetting[0])
	Contrast = "v4l2-ctl --set-ctrl contrast={}".format(cameraSetting[1])
	Saturation = "v4l2-ctl --set-ctrl saturation={}".format(cameraSetting[2])
	White_balance_temperature_auto = "v4l2-ctl --set-ctrl white_balance_temperature_auto={}".format(cameraSetting[3])
	White_balance_temperature = "v4l2-ctl --set-ctrl white_balance_temperature={}".format(cameraSetting[4])
	Sharpness = "v4l2-ctl --set-ctrl sharpness={}".format(cameraSetting[5])
	Exposure_auto = "v4l2-ctl --set-ctrl exposure_auto={}".format(cameraSetting[6])
	Exposure_absolute = "v4l2-ctl --set-ctrl exposure_absolute={}".format(cameraSetting[7])
	Exposure_auto_priority = "v4l2-ctl --set-ctrl exposure_auto_priority={}".format(cameraSetting[8])
	Focus_auto = "v4l2-ctl --set-ctrl focus_auto={}".format(cameraSetting[9])

	os.system(Brightness)
	os.system(Contrast)
	os.system(Saturation)
	os.system(White_balance_temperature_auto)
	os.system(White_balance_temperature)
	os.system(Sharpness)
	os.system(Exposure_auto)
	#os.system(Exposure_absolute)
	os.system(Exposure_auto_priority)
	os.system(Focus_auto)

	print "After Set Camera Setting Parameter"
	os.system("v4l2-ctl --list-ctrls")

def nothing(x):
	pass

def midPoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def orderPoints(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(goal_top_left, goal_bottom_left) = leftMost
	D = dist.cdist(goal_top_left[np.newaxis], rightMost, "euclidean")[0]
	(goal_bottom_right, goal_top_right) = rightMost[np.argsort(D)[::-1], :]
	return np.array([goal_top_left, goal_top_right, goal_bottom_right, goal_bottom_left], dtype="float32")

#########################################################
# New Function
# Will be implemented in version 3.0
# Will be moved to Cython to speed up for-loop iteration
#########################################################

def transToImgFrame(x,y):
	# x = x + 160
	x = x + HALF_IMAGE_WIDTH
	y = (y-IMAGE_HEIGHT) * -1
	return (x,y)

# Polar to cartesian
def polToCart(radius, theta):
	x = int(radius * math.cos(math.radians(theta)))
	y = int(radius * math.sin(math.radians(theta)))
	return (x,y)

def distancePointToPoint(p0, p1):
	#return math.sqrt(((p1[0]-p0[0])*(p1[0]-p0[0])) + ((p1[1]-p0[1])*(p1[1]-p0[1])))
	return distance.euclidean(p0, p1)

def rectToPoints(rectTopLeftX, rectTopLeftY, rectWidth, rectHeight):
	npPtRect = np.zeros((4,2), dtype=int)
	# top left		
	npPtRect[0,0] = rectTopLeftX
	npPtRect[0,1] = rectTopLeftY
	# top right		
	npPtRect[1,0] = rectTopLeftX + rectWidth
	npPtRect[1,1] = rectTopLeftY
	# bottom left		
	npPtRect[2,0] = rectTopLeftX
	npPtRect[2,1] = rectTopLeftY + rectHeight
	# bottom right		
	npPtRect[3,0] = rectTopLeftX + rectWidth
	npPtRect[3,1] = rectTopLeftY + rectHeight
	return npPtRect

# This is the next improvement
def fieldContourExtraction(inputImage, inputBinaryImage, angleStep, lengthStep, noiseThreshold, enableDebug):
	npPoint = np.zeros((1,2), dtype=int)
	outputImage = inputImage.copy()
	maxLength = int(math.sqrt(HALF_IMAGE_WIDTH*HALF_IMAGE_WIDTH+IMAGE_HEIGHT*IMAGE_HEIGHT))
	totalPoint = 0
	lastLength = 0
	angle = 180
	while angle >= 0:
		foundGreen = False
		lastFoundGreen = False
		length = maxLength
		while length >= 0:
			x,y =  polToCart(length,angle)
			x,y =  transToImgFrame(x,y)
			if x >= 0 and x < IMAGE_WIDTH and y >= 0 and y < IMAGE_HEIGHT:          
				if foundGreen == False:
					warna = inputBinaryImage.item(y,x)
			else:
				warna = 0

			# Jika belum ketemu hijau
			if foundGreen == False:
				# Ketemu warna hijau
				if warna == 255:
					arrayLength, _ = npPoint.shape
					if arrayLength >= 2:
						deltaLength = distancePointToPoint(npPoint[-2,:2], [x,y])
					else:
						deltaLength = 0
					if angle == 179:
						foundGreen = True
					else:
						if deltaLength < noiseThreshold:
							foundGreen = True
					# Update gambar dan np point
					if foundGreen == True:
						npPoint = np.insert(npPoint, totalPoint, [x,y],axis=0)
						if enableDebug == True:
							color = (0,0,255)
							cv2.circle(outputImage,(x,y), 2, color, -1)
						totalPoint += 1
						foundGreen == True  
						lastLength = length
			else:
				break
			lastFoundGreen = foundGreen
			length = length - lengthStep
		angle = angle - angleStep
	# Delete point yang terakhir
	# tambahkan point 0 dan point akhir
	#npPoint = np.insert(npPoint, 0, [0,IMAGE_HEIGHT-1],axis=0)
	#npPoint = np.insert(npPoint, -1, [IMAGE_WIDTH-1,IMAGE_HEIGHT-1],axis=0)
	npPoint = np.delete(npPoint, -1, axis=0)
	if enableDebug == True:
		plt.imshow(outputImage)
		plt.title('Output Image')
		plt.show()
	return npPoint

# Gamma Correction
# Ref : https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def gammaCorrection(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

# Distance ----------------------------------------------------------------------
def distance_to_camera(knownWidth, focalLength, perWidth):
	try:
	    return int((knownWidth * focalLength) / perWidth)
	except:
		return 0
#-------------------------------------------------------------------------------

def main():
	# print 'Number of arguments:', len(sys.argv), 'arguments.'
	# print 'Argument List:', str(sys.argv)
	# sys.exit()
	# Running Mode
	# 0 : Running Program
	# 1 : Test Dataset
	# 2 : Train Ball
	# 3 : Train Goal
	# 4 : Generate Image
	# 5 : Running With Browser Streaming

	runningMode = 5
	useMachineLearning = False

	# Machine learning model will be saved to this file
	# Declare the decission tree classifier
	ballMLModel = tree.DecisionTreeClassifier()
	goalMLModel = tree.DecisionTreeClassifier()
	# print 'oke'
	# Nanti didefinisikan di global ya
	contourColor = (0, 255, 0)
	ballColor = (0, 0, 255)
	goalColor = (255, 0, 0 )
	npBallDataset = np.zeros((1,13))
	npGoalDataset = np.zeros((1,12))
	ballProperties = np.zeros((1,11))
	goalProperties = np.zeros((1,10))

	imageNumber = 2762 # 171 #67
	ballDataNumber = 1
	goalDataNumber = 1
	ballNumber = 0
	goalNumber = 0
	ballContourLen = np.zeros(3, dtype=int)
	goalContourLen = 0

	ballMode = 0
	frameId = 2886 #data gambar terakhir di folder Learning_Image
	clear = 0

	last_debug = imageToDisplay = debug_ballmode1 = debug_ballmode2 = debug_goal = 0
	mod1 = mod2 = False

	maxRad = 0
	fieldPix0 = 0

	showHelp()
	loadConfig()
		
	if runningMode == 0 or runningMode == 5:
		print 'Running From Live Cam'
		# Open Camera
		cap = cv2.VideoCapture(0)
		# Program run from live camera
		# load machine learning model from file
		if useMachineLearning:
			ballMLModel = joblib.load(ballMLFilename)
			goalMLModel = joblib.load(goalMLFilename)
		os.system("v4l2-ctl --set-ctrl exposure_auto={}".format(3))
	elif runningMode == 1:
		# Program test mlModel from image
		if useMachineLearning:
			ballMLModel = joblib.load(ballMLFilename)
			goalMLModel = joblib.load(goalMLFilename)
		print 'Test Dataset'
	elif runningMode == 2:
		print 'Train Ball Dataset'
	elif runningMode == 3:
		print 'Train Goal Dataset'
	elif runningMode == 4:
		print 'Generate Image Dataset'

	# Image yang akan ditampilkan
	imageToDisplay = 0
	kernel = np.ones((5,5),np.uint8)

	# Connect to localhost
	try:
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	except socket.error:
		print 'Failed to create socket'
		sys.exit()

	# print os.system("ls")
	
	runningIteration = 0
	# Run streaming
	
	#init calibration single camera
	B_KNOWN_PIXEL = 104 #lebar pixel without calibration
	B_KNOWN_DISTANCE = 70 #jarak cm
	B_KNOWN_WIDTH = 14 #diameter cm
	B_focalLength = (B_KNOWN_PIXEL * B_KNOWN_DISTANCE) / B_KNOWN_WIDTH
	#LH=198:440, RH=186:460
	G_KNOWN_PIXEL = 198 #lebar pixel without calibration
	G_KNOWN_DISTANCE = 440 #jarak cm
	G_KNOWN_HEIGHT = 162 #180 #tinggi cm
	G_focalLength = (G_KNOWN_PIXEL * G_KNOWN_DISTANCE) / G_KNOWN_HEIGHT

	#calibration data
	REMAP_INTERPOLATION = cv2.INTER_LINEAR
	with np.load("/home/eko_rudiawan/BarelangFC-Vision/Calibration/B.npz") as X:
		mapx, mapy = [X[i] for i in ("mapx","mapy")]
	
	while(True):
		# Ini nanti diganti dengan load dari file
		# Create trackbar		
		if runningMode == 0 or runningMode == 4 or runningMode == 5:
			_, image = cap.read()
			# with calibrating image
			# rgbImage = cv2.remap(image, mapx, mapy, REMAP_INTERPOLATION)
			# without calibrating image
			rgbImage = cv2.resize(image, (IMAGE_WIDTH,IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

		elif runningMode == 1 or runningMode == 2 or runningMode == 3:
			# Linux
			#rawImage = '/home/eko_rudiawan/dataset/gambar_normal_' + str(imageNumber) + '.jpg'
			rawImage = '../Learning_Image/gambar_normal_' + str(imageNumber) + '.jpg'
			# Windows
			#rawImage = 'D:/RoboCupDataset/normal/gambar_normal_' + str(imageNumber) + '.jpg'
			#print rawImage
			rgbImage = cv2.imread(rawImage)

		# ini gak bagus harusnya deklarasi diatas
		fieldMask = np.zeros(rgbImage.shape[:2], np.uint8)
		notFieldMask = 255 * np.ones(rgbImage.shape[:2], np.uint8)
		
		# Color Conversion
		modRgbImage = rgbImage.copy()
		hsvImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2HSV)
		yuvImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2YUV)
		blurRgbImage = cv2.GaussianBlur(rgbImage,(5,5),0)
		grayscaleImage = cv2.cvtColor(blurRgbImage, cv2.COLOR_BGR2GRAY)
		#hsvBlurImage = cv2.cvtColor(blurRgbImage, cv2.COLOR_BGR2HSV)
		hsvBlurImage = cv2.cvtColor(blurRgbImage, cv2.COLOR_BGR2LAB)

		# Check Pixel Value
		# rMax = 10
		# averagePix_B = sumPix_B = totalPix_B = 0.0
		# averagePix_G = sumPix_G = totalPix_G = 0.0
		# averagePix_R = sumPix_R = totalPix_R = 0.0
		# for x in xrange((IMAGE_WIDTH/2)-rMax, (IMAGE_WIDTH/2)+rMax):
		# 	for y in xrange((IMAGE_HEIGHT/2)-rMax, (IMAGE_HEIGHT/2)+rMax):
		# 		pixValue_B = rgbImage[y,x,0]
		# 		pixValue_G = rgbImage[y,x,1]
		# 		pixValue_R = rgbImage[y,x,2]
		# 		sumPix_B = sumPix_B + pixValue_B
		# 		sumPix_G = sumPix_G + pixValue_G
		# 		sumPix_R = sumPix_R + pixValue_R
		# 		totalPix_B = totalPix_B + 1
		# 		totalPix_G = totalPix_G + 1
		# 		totalPix_R = totalPix_R + 1
		# averagePix_B = (float(sumPix_B)/float(totalPix_B))
		# averagePix_G = (float(sumPix_G)/float(totalPix_G))
		# averagePix_R = (float(sumPix_R)/float(totalPix_R))
		#if clear == 0 and (runningMode == 0 or runningMode == 5):
		#	cv2.rectangle(modRgbImage, ((IMAGE_WIDTH/2)-rMax,(IMAGE_HEIGHT/2)-rMax), ((IMAGE_WIDTH/2)+rMax,(IMAGE_HEIGHT/2)+rMax), (0,255,0), 3)

		# Field Green Color Filtering
		fieldGrBinary = cv2.inRange(hsvBlurImage, lowerFieldGr, upperFieldGr)
		fieldGrBinaryErode = cv2.erode(fieldGrBinary, kernel, iterations = edFieldGr[0])
 		fieldGrFinal = cv2.dilate(fieldGrBinaryErode, kernel, iterations = edFieldGr[1])

		# Field Contour Detection
		_, listFieldContours, _ = cv2.findContours(fieldGrFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if len(listFieldContours) > 0:
			fieldContours = sorted(listFieldContours, key=cv2.contourArea, reverse=True)[:1]
			if clear == 0:
				cv2.drawContours(modRgbImage,[fieldContours[0]],0,(255,255,0),2, offset=(0,0))
			cv2.drawContours(fieldMask, [fieldContours[0]], -1, 255, -1)
			cv2.drawContours(notFieldMask, [fieldContours[0]], -1, 0, -1)

		# Ball Green Filtering
		ballGrBinary = cv2.inRange(hsvImage, lowerBallGr, upperBallGr)
		ballGrBinaryInvert = cv2.bitwise_not(ballGrBinary)

		ballGrBinaryErode = cv2.erode(ballGrBinaryInvert,kernel,iterations = edBallGr[0])
 		ballGrBinaryDilate = cv2.dilate(ballGrBinaryErode,kernel,iterations = edBallGr[1])
		ballGrFinal = cv2.bitwise_and(ballGrBinaryDilate, fieldMask)

		# Ball White Filtering
		ballWhBinary = cv2.inRange(yuvImage, lowerBallWh, upperBallWh)
		#ballWhBinary = cv2.inRange(hsvImage, lowerBallWh, upperBallWh)

		ballWhBinaryErode = cv2.erode(ballWhBinary,kernel,iterations = edBallWh[0])
		ballWhBinaryDilate = cv2.dilate(ballWhBinaryErode,kernel,iterations = edBallWh[1])
		ballWhFinal = cv2.bitwise_and(ballWhBinaryDilate, fieldMask)

		# Ball Detection
		ballFound = False
		ballContourLen[0] = 0
		ballContourLen[1] = 0
		ballContourLen[2] = 0
		ballIteration = 0

		# Initialize to default
		detectedBall[0] = -1 #-888
		detectedBall[1] = -1
		detectedBall[2] = 0
		detectedBall[3] = 0
		detectedBall[4] = -1

		for ballDetectionMode in range(0, 2):
			if ballDetectionMode == 0:
			#if (ballDetectionMode == 0 and ((mod1 == False and mod2 == False) or (mod1 == True and mod2 == False))):
				_, listBallContours, _ = cv2.findContours(ballGrFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			elif ballDetectionMode == 1:
			#elif (ballDetectionMode == 1 and ((mod1 == False and mod2 == False) or (mod1 == False and mod2 == True))):
				_, listBallContours, _ = cv2.findContours(ballWhFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			if len(listBallContours) > 0:
				listSortedBallContours = sorted(listBallContours, key=cv2.contourArea, reverse=True)[:5]		
				ballContourLen[ballDetectionMode] = len(listSortedBallContours)
				ballContourLen[2] = ballContourLen[0] + ballContourLen[1]
				ball_iteration = 1
				for ballContour in listSortedBallContours:
					ballTopLeftX, ballTopLeftY, ballWidth, ballHeight = cv2.boundingRect(ballContour)
					# Program running normal
					if runningMode == 0 or runningMode == 1 or runningMode == 5:
								
						# Load model from file and run the algorithm with the model
						# Get contour properties
						# Machine learning parameter
						ballMode = ballDetectionMode

						""" 
						# mode baru
						ballArea = float(cv2.contourArea(ballContour)) / float(IMAGE_AREA)			#area
						ballRectArea = (float(ballWidth) * float(ballHeight)) / float(IMAGE_AREA)		#rect_area
						# Extent is the ratio of contour area to bounding rectangle area.
						ballExtent = float(ballArea) / float(ballRectArea)					#area_ratio
						# Aspect Ratio is the ratio of width to height of bounding rect of the object.
						ballAspectRatio = float(ballWidth) / float(ballHeight)					#wh_ratio
						# Solidity is the ratio of contour area to its convex hull area.
						ballHull = cv2.convexHull(ballContour)
						ballHullArea = cv2.contourArea(ballHull) / float(IMAGE_AREA)
						if ballHullArea > 0:
							ballSolidity = float(ballArea) / float(ballHullArea)				#percent_white
						else:
							ballSolidity = 0
						""" 

						# mode Lama
						ballArea = float(cv2.contourArea(ballContour)) / float(IMAGE_AREA) * 100.0		#area
						ballRectArea = (float(ballWidth) * float(ballHeight)) / float(IMAGE_AREA) * 100.0	#rect_area
						if ballRectArea != 0:
							""" Extent is the ratio of contour area to bounding rectangle area """
							ballExtent = float(ballArea) / float(ballRectArea)				#area_ratio
						if ballHeight != 0:
							""" Aspect Ratio is the ratio of width to height of bounding rect of the object """
							ballAspectRatio = float(ballWidth) / float(ballHeight)				#wh_ratio
						ballHull = cv2.convexHull(ballContour)
						ballHullArea = cv2.contourArea(ballHull) / float(IMAGE_AREA)
						if ballHullArea > 0:
							""" Solidity is the ratio of contour area to its convex hull area """
							ballSolidity = float(ballArea) / float(ballHullArea)				#percent_white
						else:
							ballSolidity = 0

						ballRoi = grayscaleImage[ballTopLeftY:ballTopLeftY + ballHeight, ballTopLeftX:ballTopLeftX + ballWidth]
						ballHistogram0, ballHistogram1, ballHistogram2, ballHistogram3, ballHistogram4 = cv2.calcHist([ballRoi],[0],None,[5],[0,256])
						# Rescaling to percent
						sumBallHistogram = float(ballHistogram0[0] + ballHistogram1[0] + ballHistogram2[0] + ballHistogram3[0] + ballHistogram4[0])
						ballHistogram0[0] = float(ballHistogram0[0]) / sumBallHistogram
						ballHistogram1[0] = float(ballHistogram1[0]) / sumBallHistogram
						ballHistogram2[0] = float(ballHistogram2[0]) / sumBallHistogram
						ballHistogram3[0] = float(ballHistogram3[0]) / sumBallHistogram
						ballHistogram4[0] = float(ballHistogram4[0]) / sumBallHistogram
						ballParameter = np.array([ballAspectRatio, ballArea, ballRectArea, ballExtent, ballSolidity, ballHistogram0[0], ballHistogram1[0], ballHistogram2[0], ballHistogram3[0], ballHistogram4[0], ballMode])
						ballProperties = np.insert(ballProperties, 0, ballParameter , axis = 0)
						ballProperties = np.delete(ballProperties, -1, axis=0)
						
						# print ballPrediction
						# Yes, it is a ball

						# get field pixel value for handling ball detection
						maxRad = ballWidth/4
						if maxRad > 15:
							maxRad = 15

						if (ballTopLeftY+(ballHeight/2)-((ballWidth/2)+maxRad) > 0) and (ballTopLeftY+(ballHeight/2)-((ballWidth/2)+maxRad) < IMAGE_HEIGHT):
							fieldPix0 = fieldGrFinal.copy()[(ballTopLeftY+(ballHeight/2))-((ballWidth/2)+maxRad), ballTopLeftX+(ballWidth/2)] #[Y,X]
						else:
							fieldPix0 = 0
						#print fieldPix0
								

						#useMachineLearning = False
 						if useMachineLearning == True:
							ballPrediction = ballMLModel.predict_proba(ballProperties)
 							if ballPrediction[0,1] == 1:
								# Set variable to skip next step						
								if clear == 0:
									if ballDetectionMode == 0:
										cv2.circle(modRgbImage, (int(ballTopLeftX + (ballWidth/2)), int(ballTopLeftY + (ballHeight/2))), 6, (255,255,255), -1)
									elif ballDetectionMode == 1:
										cv2.circle(modRgbImage, (int(ballTopLeftX + (ballWidth/2)), int(ballTopLeftY + (ballHeight/2))), 6, (0,0,0), -1)
									cv2.rectangle(modRgbImage, (ballTopLeftX, ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), ballColor, 2)
								detectedBall[0] = ballTopLeftX + ballWidth / 2 # Centre X
								detectedBall[1] = ballTopLeftY + ballHeight / 2 # Centre Y
								detectedBall[2] = ballWidth
								detectedBall[3] = ballHeight
								ballDistance = distance_to_camera(B_KNOWN_WIDTH, B_focalLength, ballWidth)
								detectedBall[4] = ballDistance

								ballFound = True
								break
						else:
							if (ballDetectionMode == 0 and ((mod1 == False and mod2 == False) or (mod1 == True and mod2 == False))):
								if debug_ballmode1 == 1:
									selected_ball = cv2.getTrackbarPos('Ball','Control2')
									if selected_ball == ball_iteration:
										#print 'BallMode --> Radius = %d, Area = %.2f,  Area_Rat = %.2f,  WH_Rat = %.2f,  Percent_Wh = %.2f'%(ballWidth/2, ballArea*100, ballExtent*100, ballAspectRatio*10, ballSolidity)
										#print 'Ball - H = %d, W = %d, A = %.2f, RA = %.2f, AR = %.2f, WH_Rat = %.2f, Percent_Wh = %.2f'%(ballHeight, ballWidth, ballArea, ballRectArea, ballExtent, ballAspectRatio, ballSolidity)
										print 'Ball - H = %d, W = %d, A = %.2f, RA = %.2f, AR = %.2f, WH_Rat = %.2f, Percent_Wh = %.2f'%(ballHeight, ballWidth, ballArea*100, ballRectArea, ballExtent*100, ballAspectRatio*10, ballSolidity)
										ball_color = (244, 66, 66)
									else:
										ball_color = (31,127,255)
									cv2.rectangle(modRgbImage, (ballTopLeftX, ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), ball_color, 3)

								if (ballArea >= (float(debugBallGr[3])/100)) and (ballArea <= (float(debugBallGr[2])/100)): #0-30
									if (ballExtent >= (float(debugBallGr[5])/100)) and (ballExtent <= (float(debugBallGr[4])/100)): #0-1
										if (ballAspectRatio >= (float(debugBallGr[7])/10)) and (ballAspectRatio <= (float(debugBallGr[6])/10)): #0-2
											if ballSolidity >= debugBallGr[9] and ballSolidity <= debugBallGr[8]: #0-10
												if ballWidth/2 >= debugBallGr[1] and ballWidth/2 <= debugBallGr[0]: #0-320
													if clear == 0:
														cv2.circle(modRgbImage, (int(ballTopLeftX+(ballWidth/2)), int(ballTopLeftY+(ballHeight/2)-((ballWidth/2)+maxRad))), 5, (0,0,155), -1) #X,Y
													# handling ball
													if fieldPix0 != 0:
														if clear == 0:
															cv2.circle(modRgbImage, (int(ballTopLeftX + (ballWidth/2)), int(ballTopLeftY + (ballHeight/2))), ballWidth/2, (255,255,255), 3)
															cv2.circle(modRgbImage, (int(ballTopLeftX + (ballWidth/2)), int(ballTopLeftY + (ballHeight/2))), 5, (0,0,0), -1)
															#cv2.rectangle(modRgbImage, (ballTopLeftX, ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), ballColor, 2)
														detectedBall[0] = ballTopLeftX + ballWidth / 2 # Centre X
														detectedBall[1] = ballTopLeftY + ballHeight / 2 # Centre Y
														detectedBall[2] = ballWidth
														detectedBall[3] = ballHeight
														ballDistance = distance_to_camera(B_KNOWN_WIDTH, B_focalLength, ballWidth)
														detectedBall[4] = ballDistance

														ballFound = True
														break
							elif (ballDetectionMode == 1 and ((mod1 == False and mod2 == False) or (mod1 == False and mod2 == True))):
								if debug_ballmode2 == 1:
									selected_ball = cv2.getTrackbarPos('Ball','Control2')
									if selected_ball == ball_iteration:
										#print 'BallMode --> Radius = %d, Area = %.2f,  Area_Rat = %.2f,  WH_Rat = %.2f,  Percent_Wh = %.2f'%(ballWidth/2, ballArea*100, ballExtent*100, ballAspectRatio*10, ballSolidity)
										#print 'Ball - H = %d, W = %d, A = %.2f, RA = %.2f, AR = %.2f, WH_Rat = %.2f, Percent_Wh = %.2f'%(ballHeight, ballWidth, ballArea, ballRectArea, ballExtent, ballAspectRatio, ballSolidity)
										print 'Ball - H = %d, W = %d, A = %.2f, RA = %.2f, AR = %.2f, WH_Rat = %.2f, Percent_Wh = %.2f'%(ballHeight, ballWidth, ballArea*100, ballRectArea, ballExtent*100, ballAspectRatio*10, ballSolidity)
										ball_color = (244, 66, 66)
									else:
										ball_color = (31,127,255)
									cv2.rectangle(modRgbImage, (ballTopLeftX, ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), ball_color, 3)

								if (ballArea >= (float(debugBallWh[3])/100)) and (ballArea <= (float(debugBallWh[2])/100)): #0-30
									if (ballExtent >= (float(debugBallWh[5])/100)) and (ballExtent <= (float(debugBallWh[4])/100)): #0-1
										if (ballAspectRatio >= (float(debugBallWh[7])/10)) and (ballAspectRatio <= (float(debugBallWh[6])/10)): #0-2
											if ballSolidity >= debugBallWh[9] and ballSolidity <= debugBallWh[8]: #0-10
												if ballWidth/2 >= debugBallWh[1] and ballWidth/2 <= debugBallWh[0]: #0-320
													if clear == 0:
														cv2.circle(modRgbImage, (int(ballTopLeftX+(ballWidth/2)), int(ballTopLeftY+(ballHeight/2)-((ballWidth/2)+maxRad))), 5, (0,0,155), -1) #X,Y
													# handling ball
													if fieldPix0 != 0:
														if clear == 0:
															cv2.circle(modRgbImage, (int(ballTopLeftX + (ballWidth/2)), int(ballTopLeftY + (ballHeight/2))), ballWidth/2, (0,0,0), 3)
															cv2.circle(modRgbImage, (int(ballTopLeftX + (ballWidth/2)), int(ballTopLeftY + (ballHeight/2))), 5, (255,255,255), -1)
															#cv2.rectangle(modRgbImage, (ballTopLeftX, ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), ballColor, 2)
														detectedBall[0] = ballTopLeftX + ballWidth / 2 # Centre X
														detectedBall[1] = ballTopLeftY + ballHeight / 2 # Centre Y
														detectedBall[2] = ballWidth
														detectedBall[3] = ballHeight
														ballDistance = distance_to_camera(B_KNOWN_WIDTH, B_focalLength, ballWidth)
														detectedBall[4] = ballDistance

														ballFound = True
														break
															
					elif runningMode == 2:
						# print ballIteration
						# print ballNumber
						if ballNumber == ballIteration:
							# Load model from file and run the algorithm with the model
							# Get contour properties
							# Machine learning parameter
							ballMode = ballDetectionMode

							""" 
							# mode baru
							ballArea = float(cv2.contourArea(ballContour)) / float(IMAGE_AREA)			#area
							ballRectArea = (float(ballWidth) * float(ballHeight)) / float(IMAGE_AREA)		#rect_area
							# Extent is the ratio of contour area to bounding rectangle area.
							ballExtent = float(ballArea) / float(ballRectArea)					#area_ratio
							# Aspect Ratio is the ratio of width to height of bounding rect of the object.
							ballAspectRatio = float(ballWidth) / float(ballHeight)					#wh_ratio
							# Solidity is the ratio of contour area to its convex hull area.
							ballHull = cv2.convexHull(ballContour)
							ballHullArea = cv2.contourArea(ballHull) / float(IMAGE_AREA)
							if ballHullArea > 0:
								ballSolidity = float(ballArea) / float(ballHullArea)				#percent_white
							else:
								ballSolidity = 0
							""" 

							# mode Lama
							ballArea = float(cv2.contourArea(ballContour)) / float(IMAGE_AREA) * 100.0		#area
							ballRectArea = (float(ballWidth) * float(ballHeight)) / float(IMAGE_AREA) * 100.0	#rect_area
							if ballRectArea != 0:
								""" Extent is the ratio of contour area to bounding rectangle area """
								ballExtent = float(ballArea) / float(ballRectArea)				#area_ratio
							if ballHeight != 0:
								""" Aspect Ratio is the ratio of width to height of bounding rect of the object """
								ballAspectRatio = float(ballWidth) / float(ballHeight)				#wh_ratio
							ballHull = cv2.convexHull(ballContour)
							ballHullArea = cv2.contourArea(ballHull) / float(IMAGE_AREA)
							if ballHullArea > 0:
								""" Solidity is the ratio of contour area to its convex hull area """
								ballSolidity = float(ballArea) / float(ballHullArea)				#percent_white
							else:
								ballSolidity = 0


							ballRoi = grayscaleImage[ballTopLeftY:ballTopLeftY + ballHeight, ballTopLeftX:ballTopLeftX + ballWidth]
							ballHistogram0, ballHistogram1, ballHistogram2, ballHistogram3, ballHistogram4 = cv2.calcHist([ballRoi],[0],None,[5],[0,256])
							# Rescaling to percent
							sumBallHistogram = float(ballHistogram0[0] + ballHistogram1[0] + ballHistogram2[0] + ballHistogram3[0] + ballHistogram4[0])
							ballHistogram0[0] = float(ballHistogram0[0]) / sumBallHistogram
							ballHistogram1[0] = float(ballHistogram1[0]) / sumBallHistogram
							ballHistogram2[0] = float(ballHistogram2[0]) / sumBallHistogram
							ballHistogram3[0] = float(ballHistogram3[0]) / sumBallHistogram
							ballHistogram4[0] = float(ballHistogram4[0]) / sumBallHistogram
							if clear == 0:
								cv2.rectangle(modRgbImage, (ballTopLeftX,ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), ballColor, 2)
						else:
							if clear == 0:
								# print ballContourLen
								if ballDetectionMode == 0 and ballNumber < ballContourLen[0]:
									cv2.rectangle(modRgbImage, (ballTopLeftX,ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), contourColor, 2)
								elif ballDetectionMode == 1 and ballNumber >= ballContourLen[0]:
									cv2.rectangle(modRgbImage, (ballTopLeftX,ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), contourColor, 2)
						ballIteration += 1
					ball_iteration += 1
			if ballFound == True:
				break
					
		# Goal Color Filtering
		goalWhBinary = cv2.inRange(hsvImage, lowerGoalWh, upperGoalWh)
		goalWhBinaryErode = cv2.erode(goalWhBinary,kernel,iterations = edGoalWh[0])
		goalWhBinaryDilate = cv2.dilate(goalWhBinaryErode,kernel,iterations = edGoalWh[1])
		goalWhFinal = cv2.bitwise_and(goalWhBinaryDilate,notFieldMask)

		# Goal detection variable
		goalContourLen = 0
		goalIteration = 0

		# Initialize to default
		detectedGoal[0] = -1 #-888
		detectedGoal[1] = -1
		detectedGoal[2] = 0
		detectedGoal[3] = 0
		detectedGoal[4] = 0
		detectedGoal[5] = -1
		detectedGoal[6] = -1

		# Field Contour Detection
		_, listGoalContours, _ = cv2.findContours(goalWhFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if len(listGoalContours) > 0:
			listSortedGoalContours = sorted(listGoalContours, key=cv2.contourArea, reverse=True)[:5]
			goalContourLen += len(listSortedGoalContours)
			goal_iteration = 1
			for goalContour in listSortedGoalContours:
				goalTopLeftX, goalTopLeftY, goalWidth, goalHeight = cv2.boundingRect(goalContour)
				if runningMode == 0 or runningMode == 1 or runningMode == 5:
					# mode baru
					""" 
					goalAspectRatio = float(goalWidth) / float(goalHeight)	#wh_ratio
					goalArea = float(cv2.contourArea(goalContour)) / float(IMAGE_AREA)
					goalRectArea = (float(goalWidth) * float(goalHeight)) / float(IMAGE_AREA)
					goalExtent = float(goalArea) / float(goalRectArea)	#rea_ratio
					goalHull = cv2.convexHull(goalContour)
					goalHullArea = cv2.contourArea(goalHull) / float(IMAGE_AREA)
					if goalHullArea > 0:
						goalSolidity = float(goalArea) / float(goalHullArea)	#percent_white
					else:
						goalSolidity = 0
					""" 

					# mode lama
					goalArea = float(cv2.contourArea(goalContour)) / float(IMAGE_AREA) * 100.0		#area
					goalRectArea = (float(goalWidth) * float(goalHeight)) / float(IMAGE_AREA) * 100.0	#rect_area
					if goalRectArea != 0:
						goalExtent = float(goalArea) / float(goalRectArea)				#area_ratio
					if goalHeight != 0:
						goalAspectRatio = float(goalWidth) / float(goalHeight)				#wh_ratio
					goalHull = cv2.convexHull(goalContour)
					goalHullArea = cv2.contourArea(goalHull) / float(IMAGE_AREA)
					if goalHullArea > 0:
						goalSolidity = float(goalArea) / float(goalHullArea)				#percent_white
					else:
						goalSolidity = 0

					goalRoi = grayscaleImage[goalTopLeftY:goalTopLeftY + goalHeight, goalTopLeftX:goalTopLeftX + goalWidth]
					goalRoiBinary = goalWhFinal[goalTopLeftY:goalTopLeftY + goalHeight, goalTopLeftX:goalTopLeftX + goalWidth]
					goalHistogram0, goalHistogram1, goalHistogram2, goalHistogram3, goalHistogram4 = cv2.calcHist([goalRoi], [0], None, [5], [0,256])
					# Rescaling to percent
					sumGoalHistogram = float(goalHistogram0[0] + goalHistogram1[0] + goalHistogram2[0] + goalHistogram3[0] + goalHistogram4[0])
					goalHistogram0[0] = float(goalHistogram0[0]) / sumGoalHistogram
					goalHistogram1[0] = float(goalHistogram1[0]) / sumGoalHistogram
					goalHistogram2[0] = float(goalHistogram2[0]) / sumGoalHistogram
					goalHistogram3[0] = float(goalHistogram3[0]) / sumGoalHistogram
					goalHistogram4[0] = float(goalHistogram4[0]) / sumGoalHistogram

					goalParameter = np.array([goalAspectRatio, goalArea, goalRectArea, goalExtent, goalSolidity, goalHistogram0[0], goalHistogram1[0], goalHistogram2[0], goalHistogram3[0], goalHistogram4[0]])
					goalProperties = np.insert(goalProperties, 0, goalParameter , axis = 0)
					goalProperties = np.delete(goalProperties, -1, axis=0)
					

					#useMachineLearning = True
 					if useMachineLearning == True:
						goalPrediction = goalMLModel.predict_proba(goalProperties)
						if goalPrediction[0,1] == 1:
							if clear == 0:
								cv2.rectangle(modRgbImage, (goalTopLeftX,goalTopLeftY), (goalTopLeftX + goalWidth, goalTopLeftY + goalHeight), goalColor, 2)
							# print 'Contour'
							# print goalContour[:,0,:]
							npGoalContourPoint = goalContour[:,0,:]
							# Create numpy ROI rectangle points
							npRoiTopLeft = np.array([[goalTopLeftX, goalTopLeftY]])
							npRoiTopRight = np.array([[(goalTopLeftX + goalWidth), goalTopLeftY]])
							npRoiBottomRight = np.array([[(goalTopLeftX + goalWidth), (goalTopLeftY + goalHeight)]])
							npRoiBottomLeft = np.array([[goalTopLeftX, (goalTopLeftY + goalHeight)]])
							# Gambar titik
							# cv2.circle(modRgbImage, tuple(npRoiTopLeft), 8, (0, 0, 255), -1)
							# cv2.circle(modRgbImage, tuple(npRoiTopRight), 8, (0, 255, 0), -1)
							# cv2.circle(modRgbImage, tuple(npRoiBottomRight), 8, (255, 0, 0), -1)
							# cv2.circle(modRgbImage, tuple(npRoiBottomLeft), 8, (255, 255, 0), -1)

							# Find minimum distance from ROI points to contour
							npDistanceResult = distance.cdist(npRoiTopLeft, npGoalContourPoint, 'euclidean')
							poleTopLeft = tuple(npGoalContourPoint[np.argmin(npDistanceResult),:])

							npDistanceResult = distance.cdist(npRoiTopRight, npGoalContourPoint, 'euclidean')
							poleTopRight = tuple(npGoalContourPoint[np.argmin(npDistanceResult),:])

							npDistanceResult = distance.cdist(npRoiBottomRight, npGoalContourPoint, 'euclidean')
							poleBottomRight = tuple(npGoalContourPoint[np.argmin(npDistanceResult),:])

							npDistanceResult = distance.cdist(npRoiBottomLeft, npGoalContourPoint, 'euclidean')
							poleBottomLeft = tuple(npGoalContourPoint[np.argmin(npDistanceResult),:])

							showPole = True
							if showPole == True:
								if clear == 0:
									cv2.circle(modRgbImage, poleBottomLeft, 8, (0, 0, 255), -1)
									cv2.circle(modRgbImage, poleTopRight, 8, (0, 255, 0), -1)
									cv2.circle(modRgbImage, poleTopLeft, 8, (255, 0, 0), -1)
									cv2.circle(modRgbImage, poleBottomRight, 8, (255, 255, 0), -1)
							poleLeftHeight = 0
							poleRightHeight = 0
							try:
								poleLeftHeight = np.linalg.norm(np.array(poleTopLeft)-np.array(poleBottomLeft))
								poleRightHeight = np.linalg.norm(np.array(poleTopRight)-np.array(poleBottomRight))
							except:
								pass

							# Pecah jadi 2 bagian
							goalRoiLeft = goalRoiBinary[0:goalHeight, 0:goalWidth/2]
							goalRoiRight = goalRoiBinary[0:goalHeight, goalWidth/2:goalWidth-1]

							# Hitung titik pusat gawang kiri
							poleMomentPosition = np.zeros((2,2), dtype=int)
							for goalPolePosition in range(0, 2):
								if goalPolePosition == 0:
									_, listGoalPoleContours, _ = cv2.findContours(goalRoiLeft.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
								elif goalPolePosition == 1:
									_, listGoalPoleContours, _ = cv2.findContours(goalRoiRight.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
								goalPoleContour = sorted(listGoalPoleContours, key=cv2.contourArea, reverse=True)[:1]
								if len(goalPoleContour) > 0:
											
									try:
										goalPoleMoment = cv2.moments(goalPoleContour[0])
									except:
										pass
									# Kurang exception div by zero
									try:
										poleMomentPosition[goalPolePosition, 0] = int(goalPoleMoment["m10"] / goalPoleMoment["m00"]) #x
										poleMomentPosition[goalPolePosition, 1] = int(goalPoleMoment["m01"] / goalPoleMoment["m00"]) #y
									except:
										pass
							# Gambar titik moment
							showMoment = False
							if showMoment == True:
								if clear == 0:
									cv2.circle(modRgbImage, (goalTopLeftX + poleMomentPosition[0,0], goalTopLeftY + poleMomentPosition[0,1]), 7, (50, 100, 255), -1)
									cv2.circle(modRgbImage, (goalTopLeftX + (goalWidth/2) + poleMomentPosition[1,0], goalTopLeftY + poleMomentPosition[1,1]), 7, (50, 100, 255), -1)

							# Cari selisih titik moment y dari tiang 1 dan tiang 2
							diffMomentPosition = poleMomentPosition[0,1] - poleMomentPosition[1,1]
							poleClass = 0
							# print diffMomentPosition
							if diffMomentPosition < -80:
								poleClass = 1
							elif diffMomentPosition > 80:
								poleClass = -1
							else:
								poleClass = 0

							detectedGoal[0] = goalTopLeftX + (goalWidth / 2) #X
							detectedGoal[1] = goalTopLeftY + (goalHeight / 2) #Y
							detectedGoal[2] = poleLeftHeight
							detectedGoal[3] = poleRightHeight
							detectedGoal[4] = poleClass #-1 0 1
							poleLeftDistance = distance_to_camera(G_KNOWN_HEIGHT, G_focalLength, poleLeftHeight)
							poleRightDistance = distance_to_camera(G_KNOWN_HEIGHT, G_focalLength, poleRightHeight)
							detectedGoal[5] = poleLeftDistance
							detectedGoal[6] = poleRightDistance
							# Udah ketemu ya break aja
							break
					else:
						#sepertinya goalRrctArea perlu di kali 100, cz beda dengan nilai yg lama
						if debug_goal == 1:
							selected_goal = cv2.getTrackbarPos('Goal','Control2')
							if selected_goal == goal_iteration :
								print 'Goal --> Rect_Area = %.2f,  Area_Ratio = %.2f'%(goalRectArea, goalExtent*100)
								goal_color = (0, 0, 255)
							else:
								goal_color = (255, 255, 255)
							#cv2.drawContours(modRgbImage, [goal_box], 0, goal_color, 3)
							cv2.rectangle(modRgbImage, (goalTopLeftX,goalTopLeftY), (goalTopLeftX + goalWidth, goalTopLeftY + goalHeight), goal_color, 3)

						if goalRectArea >= debugGoalWh[1] and goalRectArea <= debugGoalWh[0]:
							if (goalExtent >= (float(debugGoalWh[3])/100)) and (goalExtent <= (float(debugGoalWh[2])/100)):
								centre_color = fieldGrFinal.copy()[int(goalTopLeftY+(goalHeight/2)), int(goalTopLeftX+(goalWidth/2))]
								#print centre_color
								if centre_color == 255:
									continue # Skip kalau centre gawang ada di lapangan
								else:
									if clear == 0:
										cv2.rectangle(modRgbImage, (goalTopLeftX,goalTopLeftY), (goalTopLeftX + goalWidth, goalTopLeftY + goalHeight), goalColor, 2)
									# print 'Contour'
									# print goalContour[:,0,:]
									npGoalContourPoint = goalContour[:,0,:]
									# Create numpy ROI rectangle points
									npRoiTopLeft = np.array([[goalTopLeftX, goalTopLeftY]])
									npRoiTopRight = np.array([[(goalTopLeftX + goalWidth), goalTopLeftY]])
									npRoiBottomRight = np.array([[(goalTopLeftX + goalWidth), (goalTopLeftY + goalHeight)]])
									npRoiBottomLeft = np.array([[goalTopLeftX, (goalTopLeftY + goalHeight)]])
									# Gambar titik
									# cv2.circle(modRgbImage, tuple(npRoiTopLeft), 8, (0, 0, 255), -1)
									# cv2.circle(modRgbImage, tuple(npRoiTopRight), 8, (0, 255, 0), -1)
									# cv2.circle(modRgbImage, tuple(npRoiBottomRight), 8, (255, 0, 0), -1)
									# cv2.circle(modRgbImage, tuple(npRoiBottomLeft), 8, (255, 255, 0), -1)

									# Find minimum distance from ROI points to contour
									npDistanceResult = distance.cdist(npRoiTopLeft, npGoalContourPoint, 'euclidean')
									poleTopLeft = tuple(npGoalContourPoint[np.argmin(npDistanceResult),:])

									npDistanceResult = distance.cdist(npRoiTopRight, npGoalContourPoint, 'euclidean')
									poleTopRight = tuple(npGoalContourPoint[np.argmin(npDistanceResult),:])

									npDistanceResult = distance.cdist(npRoiBottomRight, npGoalContourPoint, 'euclidean')
									poleBottomRight = tuple(npGoalContourPoint[np.argmin(npDistanceResult),:])

									npDistanceResult = distance.cdist(npRoiBottomLeft, npGoalContourPoint, 'euclidean')
									poleBottomLeft = tuple(npGoalContourPoint[np.argmin(npDistanceResult),:])

									showPole = True
									if showPole == True:
										if clear == 0:
											cv2.circle(modRgbImage, poleBottomLeft, 8, (0, 0, 255), -1)
											cv2.circle(modRgbImage, poleTopRight, 8, (0, 255, 0), -1)
											cv2.circle(modRgbImage, poleTopLeft, 8, (255, 0, 0), -1)
											cv2.circle(modRgbImage, poleBottomRight, 8, (255, 255, 0), -1)
									poleLeftHeight = 0
									poleRightHeight = 0
									try:
										poleLeftHeight = np.linalg.norm(np.array(poleTopLeft)-np.array(poleBottomLeft))
										poleRightHeight = np.linalg.norm(np.array(poleTopRight)-np.array(poleBottomRight))
									except:
										continue

									# Pecah jadi 2 bagian
									goalRoiLeft = goalRoiBinary[0:goalHeight, 0:goalWidth/2]
									goalRoiRight = goalRoiBinary[0:goalHeight, goalWidth/2:goalWidth-1]

									# Hitung titik pusat gawang kiri
									poleMomentPosition = np.zeros((2,2), dtype=int)
									for goalPolePosition in range(0, 2):
										if goalPolePosition == 0:
											_, listGoalPoleContours, _ = cv2.findContours(goalRoiLeft.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
										elif goalPolePosition == 1:
											_, listGoalPoleContours, _ = cv2.findContours(goalRoiRight.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
										goalPoleContour = sorted(listGoalPoleContours, key=cv2.contourArea, reverse=True)[:1]
												
										try:
											goalPoleMoment = cv2.moments(goalPoleContour[0])
										except:
											pass
										# Kurang exception div by zero
										try:
											poleMomentPosition[goalPolePosition, 0] = int(goalPoleMoment["m10"] / goalPoleMoment["m00"]) #x
											poleMomentPosition[goalPolePosition, 1] = int(goalPoleMoment["m01"] / goalPoleMoment["m00"]) #y
										except:
											pass
									# Gambar titik moment
									showMoment = False
									if showMoment == True:
										if clear == 0:
											cv2.circle(modRgbImage, (goalTopLeftX + poleMomentPosition[0,0], goalTopLeftY + poleMomentPosition[0,1]), 7, (50, 100, 255), -1)
											cv2.circle(modRgbImage, (goalTopLeftX + goalWidth/2 + poleMomentPosition[1,0], goalTopLeftY + poleMomentPosition[1,1]), 7, (50, 100, 255), -1)

									# Cari selisih titik moment y dari tiang 1 dan tiang 2
									diffMomentPosition = poleMomentPosition[0,1] - poleMomentPosition[1,1]
									poleClass = 0
									# print diffMomentPosition
									if diffMomentPosition < -80:
										poleClass = 1
									elif diffMomentPosition > 80:
										poleClass = -1
									else:
										poleClass = 0

									detectedGoal[0] = goalTopLeftX + goalWidth / 2 #X
									detectedGoal[1] = goalTopLeftY + goalHeight / 2 #Y
									detectedGoal[2] = poleLeftHeight
									detectedGoal[3] = poleRightHeight
									detectedGoal[4] = poleClass #-1 0 1
									poleLeftDistance = distance_to_camera(G_KNOWN_HEIGHT, G_focalLength, poleLeftHeight)
									poleRightDistance = distance_to_camera(G_KNOWN_HEIGHT, G_focalLength, poleRightHeight)
									detectedGoal[5] = poleLeftDistance
									detectedGoal[6] = poleRightDistance
									# Udah ketemu ya break aja
									break

										
				elif runningMode == 3:
					if goalNumber == goalIteration:
						goalAspectRatio = float(goalWidth) / float(goalHeight)
						goalArea = float(cv2.contourArea(goalContour)) / float(IMAGE_AREA)
						#goalRectArea = (float(goalWidth) * float(goalHeight)) / float(IMAGE_AREA)
						goalRectArea = (float(goalWidth * goalHeight)) / float(IMAGE_AREA)
						goalExtent = float(goalArea) / float(goalRectArea)
						goalHull = cv2.convexHull(goalContour)
						goalHullArea = cv2.contourArea(goalHull) / float(IMAGE_AREA)
						if goalHullArea > 0:
							goalSolidity = float(goalArea) / float(goalHullArea)
						else:
							goalSolidity = 0
						goalRoi = grayscaleImage[goalTopLeftY:goalTopLeftY + goalHeight, goalTopLeftX:goalTopLeftX + goalWidth]
						goalHistogram0, goalHistogram1, goalHistogram2, goalHistogram3, goalHistogram4 = cv2.calcHist([goalRoi], [0], None, [5], [0,256])
						sumGoalHistogram = float(goalHistogram0[0] + goalHistogram1[0] + goalHistogram2[0] + goalHistogram3[0] + goalHistogram4[0])
						goalHistogram0[0] = float(goalHistogram0[0]) / sumGoalHistogram
						goalHistogram1[0] = float(goalHistogram1[0]) / sumGoalHistogram
						goalHistogram2[0] = float(goalHistogram2[0]) / sumGoalHistogram
						goalHistogram3[0] = float(goalHistogram3[0]) / sumGoalHistogram
						goalHistogram4[0] = float(goalHistogram4[0]) / sumGoalHistogram
						if clear == 0:
							cv2.rectangle(modRgbImage, (goalTopLeftX,goalTopLeftY), (goalTopLeftX + goalWidth, goalTopLeftY + goalHeight), goalColor, 2)
					else:
						if clear == 0:
							cv2.rectangle(modRgbImage, (goalTopLeftX,goalTopLeftY), (goalTopLeftX + goalWidth, goalTopLeftY + goalHeight), contourColor, 2)
					goalIteration += 1
				goal_iteration += 1

		# Send detection result to localhost
		try:
			#s.flush()
			visionMessage = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d'%(detectedBall[0], detectedBall[1], detectedBall[2], detectedBall[3], detectedBall[4], detectedGoal[0], detectedGoal[1], detectedGoal[2], detectedGoal[3], detectedGoal[4], detectedGoal[5], detectedGoal[6])
			s.sendto(visionMessage, (host, port))
		except socket.error:
			#print 'Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
			sys.exit()

		font = cv2.FONT_HERSHEY_SIMPLEX
		# print 'masuk'
		if runningMode == 0 or runningMode == 1 or runningMode == 5:
			#textLine = "Running ==> Image : {} Ball : {} Dataset : {}".format(imageNumber, ballNumber, ballDataNumber)
			#textLine1 = 'Ball --> X = %d, Y = %d, D = %dcm'%(detectedBall[0], detectedBall[1], detectedBall[4])
			#textLine1 = 'Ball --> W = %d, H = %d'%(detectedBall[2], detectedBall[3])
			#textLine2 = 'Goal --> X = %d, Y = %d, C = %d, LD = %dcm, RD = %dcm'%(detectedGoal[0], detectedGoal[1], detectedGoal[4], detectedGoal[5], detectedGoal[6])
			#textLine2 = 'Goal --> LH = %d, RH = %d'%(detectedGoal[2], detectedGoal[3])
			#textLine3 = 'PixVelue [B,G,R] = %.f, %.f, %.f'%(averagePix_B, averagePix_G, averagePix_R)
			textLine1 = 'Bola   --> Jarak Bola = %d cm'%(detectedBall[4])
			textLine2 = 'Gawang --> Jarak Tiang Kiri = %d cm'%(detectedGoal[5])
			textLine3 = 'Gawang --> Jarak Tiang Kanan = %d cm'%(detectedGoal[6])
			if clear == 0:
				#cv2.putText(modRgbImage, textLine, (10,470), font, 0.5, (255,255,255),1,cv2.LINE_AA)
				cv2.putText(modRgbImage, textLine1, (10,430), font, 0.4, (255,255,255),1,cv2.LINE_AA)
				cv2.putText(modRgbImage, textLine2, (10,450), font, 0.4, (255,255,255),1,cv2.LINE_AA)
				cv2.putText(modRgbImage, textLine3, (10,470), font, 0.4, (255,255,255),1,cv2.LINE_AA)
		elif runningMode == 2:
			textLine = "Train Ball ==> Image : {} Ball : {} Dataset : {}".format(imageNumber, ballNumber, ballDataNumber)
			if clear == 0:
				cv2.putText(modRgbImage, textLine, (10,470), font, 0.5, (255,255,255),1,cv2.LINE_AA)
		elif runningMode == 3:
			textLine = "Train Goal ==> Image : {} Goal : {} Dataset : {}".format(imageNumber, goalNumber, goalDataNumber)
			if clear == 0:
				cv2.putText(modRgbImage, textLine, (10,470), font, 0.5, (255,255,255),1,cv2.LINE_AA)

		if imageToDisplay == 1: #field
			lowerFieldGr[0] = cv2.getTrackbarPos('HMin','Control')
			lowerFieldGr[1] = cv2.getTrackbarPos('SMin','Control')
			lowerFieldGr[2] = cv2.getTrackbarPos('VMin','Control')
			upperFieldGr[0] = cv2.getTrackbarPos('HMax','Control')
			upperFieldGr[1] = cv2.getTrackbarPos('SMax','Control')
			upperFieldGr[2] = cv2.getTrackbarPos('VMax','Control')
			edFieldGr[0] = cv2.getTrackbarPos('Erode','Control')
			edFieldGr[1] = cv2.getTrackbarPos('Dilate','Control')
			cv2.imshow("Barelang Vision", modRgbImage)
			cv2.imshow("Field Binary Image", fieldGrFinal)
		elif imageToDisplay == 2: #bola mode 1
			lowerBallGr[0] = cv2.getTrackbarPos('HMin','Control')
			lowerBallGr[1] = cv2.getTrackbarPos('SMin','Control')
			lowerBallGr[2] = cv2.getTrackbarPos('VMin','Control')
			upperBallGr[0] = cv2.getTrackbarPos('HMax','Control')
			upperBallGr[1] = cv2.getTrackbarPos('SMax','Control')
			upperBallGr[2] = cv2.getTrackbarPos('VMax','Control')
			edBallGr[0] = cv2.getTrackbarPos('Erode','Control')
			edBallGr[1] = cv2.getTrackbarPos('Dilate','Control')

			debug_ballmode1 = cv2.getTrackbarPos('Debug Ball Mode1','Control')
			if last_debug != debug_ballmode1:
				if (debug_ballmode1 == 1):
					createTrackbars(5)
					loadTrackbars(5)
				else:
					cv2.destroyWindow("Control2")
				last_debug = debug_ballmode1
			if (debug_ballmode1 == 1):
				debugBallGr[0] = cv2.getTrackbarPos('Radius Max', 'Control2')
				debugBallGr[1] = cv2.getTrackbarPos('Radius Min', 'Control2')
				debugBallGr[2] = cv2.getTrackbarPos('Area Max', 'Control2')
				debugBallGr[3] = cv2.getTrackbarPos('Area Min', 'Control2')
				debugBallGr[4] = cv2.getTrackbarPos('Area Ratio Max', 'Control2')
				debugBallGr[5] = cv2.getTrackbarPos('Area Ratio Min', 'Control2')
				debugBallGr[6] = cv2.getTrackbarPos('Wh Ratio Max', 'Control2')
				debugBallGr[7] = cv2.getTrackbarPos('Wh Ratio Min', 'Control2')
				debugBallGr[8] = cv2.getTrackbarPos('Percent White Max', 'Control2')
				debugBallGr[9] = cv2.getTrackbarPos('Percent White Min', 'Control2')

			cv2.imshow("Barelang Vision", modRgbImage)
			#cv2.imshow("Ball Mode 1 (Green) Binary Image", ballGrBinary)
			cv2.imshow("Ball Mode 1 (Green) Final Image", ballGrFinal)
		elif imageToDisplay == 3: #bola mode 2
			lowerBallWh[0] = cv2.getTrackbarPos('HMin','Control')
			lowerBallWh[1] = cv2.getTrackbarPos('SMin','Control')
			lowerBallWh[2] = cv2.getTrackbarPos('VMin','Control')
			upperBallWh[0] = cv2.getTrackbarPos('HMax','Control')
			upperBallWh[1] = cv2.getTrackbarPos('SMax','Control')
			upperBallWh[2] = cv2.getTrackbarPos('VMax','Control')
			edBallWh[0] = cv2.getTrackbarPos('Erode','Control')
			edBallWh[1] = cv2.getTrackbarPos('Dilate','Control')

			debug_ballmode2 = cv2.getTrackbarPos('Debug Ball Mode2','Control')
			if last_debug != debug_ballmode2:
				if (debug_ballmode2 == 1):
					createTrackbars(6)
					loadTrackbars(6)
				else:
					cv2.destroyWindow("Control2")
				last_debug = debug_ballmode2
			if (debug_ballmode2 == 1):
				debugBallWh[0] = cv2.getTrackbarPos('Radius Max', 'Control2')
				debugBallWh[1] = cv2.getTrackbarPos('Radius Min', 'Control2')
				debugBallWh[2] = cv2.getTrackbarPos('Area Max', 'Control2')
				debugBallWh[3] = cv2.getTrackbarPos('Area Min', 'Control2')
				debugBallWh[4] = cv2.getTrackbarPos('Area Ratio Max', 'Control2')
				debugBallWh[5] = cv2.getTrackbarPos('Area Ratio Min', 'Control2')
				debugBallWh[6] = cv2.getTrackbarPos('Wh Ratio Max', 'Control2')
				debugBallWh[7] = cv2.getTrackbarPos('Wh Ratio Min', 'Control2')
				debugBallWh[8] = cv2.getTrackbarPos('Percent White Max', 'Control2')
				debugBallWh[9] = cv2.getTrackbarPos('Percent White Min', 'Control2')

			cv2.imshow("Barelang Vision", modRgbImage)
			cv2.imshow("Ball Mode 2 (White) Binary Image", ballWhFinal)
		elif imageToDisplay == 4: #goal
			lowerGoalWh[0] = cv2.getTrackbarPos('HMin','Control')
			lowerGoalWh[1] = cv2.getTrackbarPos('SMin','Control')
			lowerGoalWh[2] = cv2.getTrackbarPos('VMin','Control')
			upperGoalWh[0] = cv2.getTrackbarPos('HMax','Control')
			upperGoalWh[1] = cv2.getTrackbarPos('SMax','Control')
			upperGoalWh[2] = cv2.getTrackbarPos('VMax','Control')
			edGoalWh[0] = cv2.getTrackbarPos('Erode','Control')
			edGoalWh[1] = cv2.getTrackbarPos('Dilate','Control')

			debug_goal = cv2.getTrackbarPos('Debug Goal','Control')
			if last_debug != debug_goal:
				if (debug_goal == 1):
					createTrackbars(7)
					loadTrackbars(7)
				else:
					cv2.destroyWindow("Control2")
				last_debug = debug_goal
			if (debug_goal == 1):
				debugGoalWh[0] = cv2.getTrackbarPos('Rect Area Max', 'Control2')
				debugGoalWh[1] = cv2.getTrackbarPos('Rect Area Min', 'Control2')
				debugGoalWh[2] = cv2.getTrackbarPos('Area Ratio Max', 'Control2')
				debugGoalWh[3] = cv2.getTrackbarPos('Area Ratio Min', 'Control2')

			cv2.imshow("Barelang Vision", modRgbImage)
			cv2.imshow("Goal Binary Image", goalWhFinal)
		elif imageToDisplay == 8: #setCameraParameter
			cameraSetting[0] = cv2.getTrackbarPos('brightness','Control')
			cameraSetting[1] = cv2.getTrackbarPos('contrast','Control')
			cameraSetting[2] = cv2.getTrackbarPos('saturation','Control')
			cameraSetting[3] = cv2.getTrackbarPos('white_balance_temperature_auto','Control')
			cameraSetting[4] = cv2.getTrackbarPos('white_balance_temperature','Control')
			if cameraSetting[4] < 2000:
				cv2.setTrackbarPos('white_balance_temperature','Control',2000)
			cameraSetting[5] = cv2.getTrackbarPos('sharpness','Control')
			cameraSetting[6] = cv2.getTrackbarPos('exposure_auto','Control')
			cameraSetting[7] = cv2.getTrackbarPos('exposure_absolute','Control')
			if cameraSetting[7] < 3:
				cv2.setTrackbarPos('exposure_absolute','Control',3)
			cameraSetting[8] = cv2.getTrackbarPos('exposure_auto_priority','Control')
			cameraSetting[9] = cv2.getTrackbarPos('focus_auto','Control')
			#setCameraParameter()
			cv2.imshow("Barelang Vision", modRgbImage)
		else:
			# Hanya tampil kalau mode stream url tdk aktif
			if runningMode != 5:
				cv2.imshow("Barelang Vision", modRgbImage)
			
		# Waiting keyboard interrupt
		k = cv2.waitKey(1)
		if k == ord('z'):
			showHelp()
			#print 'Show Help'
		elif k == ord('o'):
			cv2.destroyAllWindows()
			imageToDisplay = 8
			createTrackbars(imageToDisplay)
			loadTrackbars(imageToDisplay)
			print'set Camera Parameter'
		if k == ord('p'):
			setCameraParameter()
			#print 'Set Camera Parameter'
		if k == ord('q'):
			if clear == 0:
				clear = 1
				print 'Hidden Line'
			elif clear == 1:
				clear = 0
				print 'Show Line'
		elif k == ord('c'):
			cv2.imwrite(IMAGE_PATH.format(frameId), modRgbImage)
			frameId += 1
			print 'Capture'

		# Keyboard shortcut for running mode
		if runningMode == 0:
			if k == ord('x'):
				cv2.destroyAllWindows()
				imageToDisplay = 0
				print 'Exit Program'
				break
			elif k == ord('f'):
				cv2.destroyAllWindows()
				last_debug = 0
				imageToDisplay = 1
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Field Parameter'
			elif k == ord('n'):
				cv2.destroyAllWindows()
				last_debug = 0
				imageToDisplay = 2
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				mod1 = True
				mod2 = False
				print 'Setting Ball Green Parameter'
			elif k == ord('b'):
				cv2.destroyAllWindows()
				last_debug = 0
				imageToDisplay = 3
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				mod2 = True
				mod1 = False
				print 'Setting Ball White Parameter'
			elif k == ord('g'):
				cv2.destroyAllWindows()
				last_debug = 0
				imageToDisplay = 4
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Goal Parameter'
			elif k == ord('s'):
				saveConfig()
				print 'Save Setting Value'
			elif k == ord('l'):
				loadConfig()
				loadTrackbars(imageToDisplay)
				print 'Load Setting Value'
			elif k == ord('r'):
				cv2.destroyAllWindows()
				last_debug = imageToDisplay = debug_ballmode1 = debug_ballmode2 = debug_goal = 0
				mod1 = mod2 = False
				print 'Close All Windows'
		# Keyboard Shortcut for Dataset Testing From Image
		elif runningMode == 1:
			# print 'asdasd'
			if k == ord('x'):
				cv2.destroyAllWindows()
				imageToDisplay = 0
				print 'Exit Program'
				break
			elif k == ord('f'):
				cv2.destroyAllWindows()
				imageToDisplay = 1
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Field Parameter'
			elif k == ord('n'):
				cv2.destroyAllWindows()
				imageToDisplay = 2
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball Green Parameter'
			elif k == ord('b'):
				cv2.destroyAllWindows()
				imageToDisplay = 3
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball White Parameter'
			elif k == ord('g'):
				cv2.destroyAllWindows()
				imageToDisplay = 4
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Goal Parameter'
			elif k == ord('s'):
				saveConfig()
				print 'Save Setting Value'
			elif k == ord('l'):
				loadConfig()
				loadTrackbars(imageToDisplay)
				print 'Load Setting Value'
			elif k == ord('r'):
				cv2.destroyAllWindows()
				debugmode = 0
				imageToDisplay = 0
				print 'Close All Windows'

			elif k == ord('u'):
				imageNumber += 1
				print 'Next Image'
			elif k == ord('j'):
				imageNumber -= 1
				print 'Previous Image'
		# Keyboard shortcut for ball training mode
		elif runningMode == 2:
			if k == ord('x'):
				cv2.destroyAllWindows()
				imageToDisplay = 0
				np.savetxt(ballDatasetFilename, npBallDataset, fmt='%.5f', delimiter=',', header="Samples,  Aspect Ratio,  Area,  Rect Area, Extent,  Solidity,  H0,  H1, H2, H3, H4, Mode, Ball")
				print 'Exit Program'
				break
			elif k == ord('f'):
				cv2.destroyAllWindows()
				imageToDisplay = 1
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Field Parameter'
			elif k == ord('n'):
				cv2.destroyAllWindows()
				imageToDisplay = 2
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball Green Parameter'
			elif k == ord('b'):
				cv2.destroyAllWindows()
				imageToDisplay = 3
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball White Parameter'
			elif k == ord('g'):
				cv2.destroyAllWindows()
				imageToDisplay = 4
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Goal Parameter'
			elif k == ord('s'):
				saveConfig()
				print 'Save Setting Value'
			elif k == ord('l'):
				loadConfig()
				loadTrackbars(imageToDisplay)
				print 'Load Setting Value'
			elif k == ord('r'):
				cv2.destroyAllWindows()
				imageToDisplay = 0
				print 'Close All Windows'

			elif k == ord('u'):
				imageNumber += 1
				print 'Next Image'
			elif k == ord('j'):
				imageNumber -= 1
				print 'Previous Image'
			elif k == ord('k'):
				ballNumber += 1
				if ballNumber >= ballContourLen[2]:
					ballNumber = ballContourLen[2]
					#imageNumber += 1
				print 'Next Ball Contour'
			elif k == ord('h'):
				ballNumber -= 1
				if ballNumber <= 0:
					ballNumber = 0
					#imageNumber -= 1
				print 'Previous Ball Contour'

			elif k == ord('1'): #ball
				isBall = 1
				npBallData = np.array([ballDataNumber, ballAspectRatio, ballArea, ballRectArea, ballExtent, ballSolidity, ballHistogram0[0], ballHistogram1[0], ballHistogram2[0], ballHistogram3[0], ballHistogram4[0], ballMode, isBall])
				npBallDataset = np.insert(npBallDataset, ballDataNumber-1, npBallData, axis=0)
				ballNumber += 1
				if ballNumber >= ballContourLen[2]:
					ballNumber = 0
					imageNumber += 1
				ballDataNumber += 1
				print 'Mark as Ball'
			elif k == ord('0'): #unknown
				isBall = 0
				npBallData = np.array([ballDataNumber, ballAspectRatio, ballArea, ballRectArea, ballExtent, ballSolidity, ballHistogram0[0], ballHistogram1[0], ballHistogram2[0], ballHistogram3[0], ballHistogram4[0], ballMode, isBall])
				npBallDataset = np.insert(npBallDataset, ballDataNumber-1, npBallData, axis=0)
				ballNumber += 1
				if ballNumber >= ballContourLen[2]:
					ballNumber = 0
					imageNumber += 1
				ballDataNumber += 1
				print 'Mark as Unknown'
			elif k == ord('t'):			
				npBallDataset = np.delete(npBallDataset, -1, axis=0)
				inputBallTraining = npBallDataset[:,1:12]
				outputBallTraining = npBallDataset[:,-1]
				ballMLModel = ballMLModel.fit(inputBallTraining, outputBallTraining)
				joblib.dump(ballMLModel, ballMLFilename)
				print 'Train Ball ML Model and Save to SAV'
			elif k == ord('d'):			
				np.savetxt(ballDatasetFilename, npBallDataset, fmt='%.5f', delimiter=',', header="Samples,  Aspect Ratio,  Area,  Rect Area, Extent,  Solidity,  H0,  H1, H2, H3, H4, Mode, Ball")
				print 'Save Ball Dataset to CSV'
			elif k == ord('m'):			
				print 'Load CSV Dataset, Train and Save Model'
		# Keyboard shortcut for goal training mode
		elif runningMode == 3:
			if k == ord('x'):
				cv2.destroyAllWindows()
				imageToDisplay = 0
				np.savetxt(goalDatasetFilename, npGoalDataset, fmt='%.5f', delimiter=',', header="Samples,  Aspect Ratio,  Area,  Rect Area, Extent,  Solidity,  H0,  H1, H2, H3, H4, Goal")
				print 'Exit Program'
				break
			elif k == ord('f'):
				cv2.destroyAllWindows()
				imageToDisplay = 1
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Field Parameter'
			elif k == ord('n'):
				cv2.destroyAllWindows()
				imageToDisplay = 2
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball Green Parameter'
			elif k == ord('b'):
				cv2.destroyAllWindows()
				imageToDisplay = 3
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball White Parameter'
			elif k == ord('g'):
				cv2.destroyAllWindows()
				imageToDisplay = 4
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Goal Parameter'
			elif k == ord('s'):
				saveConfig()
				print 'Save Setting Value'
			elif k == ord('l'):
				loadConfig()
				loadTrackbars(imageToDisplay)
				print 'Load Setting Value'
			elif k == ord('r'):
				cv2.destroyAllWindows()
				imageToDisplay = 0
				print 'Close All Windows'
			elif k == ord('u'):
				imageNumber += 1
				print 'Next Image'
			elif k == ord('j'):
				imageNumber -= 1
				print 'Previous Image'
			elif k == ord('k'):
				goalNumber += 1
				if goalNumber >= goalContourLen:
					goalNumber = goalContourLen
					#imageNumber += 1
				print 'Next Goal Contour'
			elif k == ord('h'):
				goalNumber -= 1
				if goalNumber <= 0:
					goalNumber = 0
					#imageNumber -= 1
				print 'Previous Goal Contour'
			elif k == ord('1'): #goal
				isGoal = 1
				npGoalData = np.array([goalDataNumber, goalAspectRatio, goalArea, goalRectArea, goalExtent, goalSolidity, goalHistogram0[0], goalHistogram1[0], goalHistogram2[0], goalHistogram3[0], goalHistogram4[0], isGoal])
				npGoalDataset = np.insert(npGoalDataset, goalDataNumber-1, npGoalData, axis=0)
				goalNumber += 1
				if goalNumber >= goalContourLen:
					goalNumber = 0
					imageNumber += 1
				goalDataNumber += 1
				print 'Mark as Goal'
			elif k == ord('0'): #unknown
				isGoal = 0
				npGoalData = np.array([goalDataNumber, goalAspectRatio, goalArea, goalRectArea, goalExtent, goalSolidity, goalHistogram0[0], goalHistogram1[0], goalHistogram2[0], goalHistogram3[0], goalHistogram4[0], isGoal])
				npGoalDataset = np.insert(npGoalDataset, goalDataNumber-1, npGoalData, axis=0)
				goalNumber += 1
				if goalNumber >= goalContourLen:
					goalNumber = 0
					imageNumber += 1
				goalDataNumber += 1
				print 'Mark as Unknown'
			elif k == ord('t'):			
				npGoalDataset = np.delete(npGoalDataset, -1, axis=0)
				inputGoalTraining = npGoalDataset[:,1:11]
				outputGoalTraining = npGoalDataset[:,-1]
				goalMLModel = goalMLModel.fit(inputGoalTraining, outputGoalTraining)
				joblib.dump(goalMLModel, goalMLFilename)
				print 'Train Goal ML Model and Save to SAV'
			elif k == ord('d'):			
				np.savetxt(goalDatasetFilename, npGoalDataset, fmt='%.5f', delimiter=',', header="Samples,  Aspect Ratio,  Area,  Rect Area, Extent,  Solidity,  H0,  H1, H2, H3, H4, Goal")
				print 'Save Goal Dataset to CSV'
			elif k == ord('m'):			
				print 'Load CSV Goal Dataset, Train and Save Model'
		# Program save image belum ada
		elif runningMode == 4:
			print 'save image'
		elif runningMode == 5:
			cv2.imwrite('stream.jpg', modRgbImage)
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + open('stream.jpg', 'rb').read() + b'\r\n')
	
	cap.release()
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	url = "http://0.0.0.0:3333"
	if os.name == "nt":
		chromedir= 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
		webbrowser.get(chromedir).open(url)
	else:
		webbrowser.get(using='firefox').open_new_tab(url)
	app.run(host='0.0.0.0', port=3333, debug=False, threaded=False)

''' 
# Decision Tree Ball Detection
if (ball_found == False and mod1 == False and mod2 == False) or (mod1 == True and mod2 == False) :
	# ukuran bola dekat dan ukuran bola jauh
	if ball_area >= 0.01 and ball_area <= 15: # Ball ball_area maximal for Detection
		if ball_area_ratio >= 0.2:
			if ball_wh_ratio >= 0.7 and ball_wh_ratio <= 1.5: # 0.5 2.7
				if percent_white >= 4: # Ball must have minimal 50% white pixel
					if ball_radius >= 10 and ball_radius <= 108: #radius bola ukuran minimal dan maximal
						ball_found = True
						#cv2.rectangle(image, (ball_topleft_x,ball_topleft_y), (ball_topleft_x + ball_width, ball_topleft_y + ball_height), (255, 255, 255), 2)
						cv2.circle(image, (int(ball_centre_x), int(ball_centre_y)), ball_radius, (255,255,255), 3)
						cv2.circle(image, (int(ball_centre_x), int(ball_centre_y)), 5, (0,255,0), -1)
					break

# Decision Tree Ball Detection
if (ball_found == False and mod2 == False and mod1 == False) or (mod2 == True and mod1 == False) :
	# ukuran bola dekat dan ukuran bola jauh
	if ball_area >= 0.5 and ball_area <= 8: # Ball ball_area maximal for Detection
		if ball_area_ratio >= 0.35:
			if ball_wh_ratio >= 0.8 and ball_wh_ratio <= 1.65:
				if percent_white >= 4: #21: #25: #30: # Ball must have minimal 50% white pixel
					if ball_radius >= 12 and ball_radius <= 108: #radius bola ukuran minimal dan maximal
						ball_found = True
						#cv2.rectangle(image, (ball_topleft_x,ball_topleft_y), (ball_topleft_x + ball_width, ball_topleft_y + ball_height), (255, 255, 255), 2) #(244,66,66)
						cv2.circle(image, (int(ball_centre_x), int(ball_centre_y)), ball_width/2 , (0,0,0), 3)
						cv2.circle(image, (int(ball_centre_x), int(ball_centre_y)), 5, (255,255,255), -1)
						break
# Decision Tree Goal Detection
if goal_found == False:
	if goal_rect_area >= 4: #6:
		if goal_area_ratio >= 0.07 and goal_area_ratio <= 0.2 :
			centre_color = mask[int(goal_centre_y), int(goal_centre_x)]
			#print centre_color
			if centre_color == 255:
				continue # Skip kalau centre gawang ada di lapangan
			else:
			#if goal_centre_x > field_xtop and goal_centre_x < field_xbot and goal_centre_y > field_ytop and goal_centre_y < field_ybot:
			#    continue # Skip kalau centre gawang ada di lapangan
			#else:
				goal_found = True
				cv2.rectangle(image,(x_box,y_box),(x_box+w_box,y_box+h_box),(255,0,0),3)
				cv2.circle(image, (int(goal_centre_x_box), int(goal_centre_y_box)), 7, (0,0,255), -1)
				#cv2.drawContours(image,[goal_box],0,(255,0,0),3)
				#cv2.line(image, (int(goal_midtop_x), int(goal_midtop_y)), (int(goal_midbot_x), int(goal_midbot_y)),(255, 0, 255), 2)
				#cv2.line(image, (int(goal_midleft_x), int(goal_midleft_y)), (int(goal_midright_x), int(goal_midright_y)),(255, 0, 255), 2)
				#for ((x, y), color) in zip(goal_rot_rect, colors):
				#	cv2.circle(image, (int(x), int(y)), 7, color, -1)
				#	cv2.circle(image, (int(goal_centre_x), int(goal_centre_y)), 7, (31,127,255), -1)
				break
''' 