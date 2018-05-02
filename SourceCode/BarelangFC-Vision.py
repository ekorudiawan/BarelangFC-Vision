######################################################################
# BarelangFC Vision V2.0                                             #
# By : Eko Rudiawan                                                  #
#                                                                    #
# We use machine learning to solve ball and goal recognition problem #
#                                                                    #
######################################################################

# Standard imports
import os
import cv2
import numpy as np
import datetime
import socket
import sys
from scipy.spatial import distance as dist
import time
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn import tree
from sklearn.externals import joblib

# Default filename to save all data
imageDatasetPath = 'D:/RoboCupDataset/dataset_lighting/my_photo-' #'D:/RoboCupDataset/normal/gambar_normal_'
settingValueFilename = 'BarelangFC-SettingValue.csv'
ballDatasetFilename = 'BarelangFC-BallDataset.csv'
ballMLFilename = 'BarelangFC-BallMLModel.sav'
goalDatasetFilename = 'BarelangFC-GoalDataset.csv'
goalMLFilename = 'BarelangFC-GoalMLModel.sav'

# Global variable for thresholding
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

# Global variable detection result
detectedBall = np.zeros(5, dtype=int)
detectedGoal = np.zeros(7, dtype=int)

host = 'localhost'
port = 2000

def showHelp():
	print '----------BarelangFC-Vision---------------------------------'
	print '### All Running Mode ### -----------------------------------'	
	print 'Parse Field -------------------------------------------- [1]'	
	print 'Parse Ball Green (Mode 1) ------------------------------ [2]'
	print 'Parse Ball White (Mode 2) ------------------------------ [3]'		
	print 'Parse Goal --------------------------------------------- [4]'
	print 'Save Filter Config ------------------------------------- [S]'
	print 'Load Filter Config ------------------------------------- [L]'
	print 'Destroy All Windows ------------------------------------ [0]'
	print 'Help --------------------------------------------------- [H]'
	print 'Exit BarelangFC-Vision --------------------------------- [X]'
	print '### Testing Mode ### ---------------------------------------'
	print 'Next Image --------------------------------------------- [N]'
	print 'Previous Image ----------------------------------------- [P]'
	print '### Training Mode ### --------------------------------------'
	print 'Next Contour ------------------------------------------- [C]'
	print 'Previous Contour --------------------------------------- [Z]'
	print '### Ball Training Mode ### ---------------------------------'
	print 'Mark as Ball ------------------------------------------- [B]'
	print 'Mark as Not Ball --------------------------------------- [U]'
	print 'Train Ball Dataset and Save ML Model ------------------- [T]'
	print 'Save Ball Dataset to CSV ------------------------------- [D]'
	print 'Load Ball Dataset from CSV, Train and Save ML Model ---- [M]' 
	print '### Goal Training Mode ### ---------------------------------'
	print 'Mark as Goal ------------------------------------------- [G]'
	print 'Mark as Not Goal --------------------------------------- [U]'
	print 'Train Goal Dataset and Save ML Model ------------------- [T]'
	print 'Save Goal Dataset to CSV ------------------------------- [D]'
	print 'Load Goal Dataset from CSV, Train and Save ML Model ---- [M]' 

def createTrackbars(mode):
	cv2.namedWindow('Control')
	# Special for setting blur image
	# Only available for field setting 
	# All setting parameter
	cv2.createTrackbar('HMin','Control',0,255,nothing)
	cv2.createTrackbar('SMin','Control',0,255,nothing)
	cv2.createTrackbar('VMin','Control',0,255,nothing)
	cv2.createTrackbar('HMax','Control',255,255,nothing)
	cv2.createTrackbar('SMax','Control',255,255,nothing)
	cv2.createTrackbar('VMax','Control',255,255,nothing)
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
	# Show Ball Green
	elif mode == 2:
		cv2.setTrackbarPos('HMin', 'Control', lowerBallGr[0])
		cv2.setTrackbarPos('SMin', 'Control', lowerBallGr[1])
		cv2.setTrackbarPos('VMin', 'Control', lowerBallGr[2])
		cv2.setTrackbarPos('HMax', 'Control', upperBallGr[0])
		cv2.setTrackbarPos('SMax', 'Control', upperBallGr[1])
		cv2.setTrackbarPos('VMax', 'Control', upperBallGr[2])
		cv2.setTrackbarPos('Erode', 'Control', edBallGr[0])
		cv2.setTrackbarPos('Dilate', 'Control', edBallGr[1])
	# Show Ball White
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

def saveConfig():
	npSettingValue = np.zeros(32, dtype=int)
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
	npSettingValue = np.reshape(npSettingValue, (1, 32))
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
	print 'Setting Parameter Loaded'

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

# Transform scanning coordinat to camera coordinat
# Definisi umum
IMAGE_WIDTH = 640
HALF_IMAGE_WIDTH = IMAGE_WIDTH / 2
IMAGE_HEIGHT = 480

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
    return math.sqrt(((p1[0]-p0[0])*(p1[0]-p0[0])) + ((p1[1]-p0[1])*(p1[1]-p0[1])))

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
    # npPoint = np.insert(npPoint, 0, [0,IMAGE_HEIGHT-1],axis=0)
    # npPoint = np.insert(npPoint, -1, [IMAGE_WIDTH-1,IMAGE_HEIGHT-1],axis=0)
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
	runningMode = 1

	# Machine learning model will be saved to this file
	# Declare the decission tree classifier 
	ballMLModel = tree.DecisionTreeClassifier()
	goalMLModel = tree.DecisionTreeClassifier()
	# Nanti didefinisikan di global ya
	contourColor = (0, 255, 0)
	ballColor = (0, 0, 255)
	goalColor = (255, 0, 0 )
	npBallDataset = np.zeros((1,13))
	npGoalDataset = np.zeros((1,12))
	ballProperties = np.zeros((1,11))
	goalProperties = np.zeros((1,10))

	imageNumber = 5 # 171 #67
	ballDataNumber = 1
	goalDataNumber = 1
	ballNumber = 0
	goalNumber = 0
	ballContourLen = np.zeros(3, dtype=int)
	goalContourLen = 0

	ballMode = 0
	
	loadConfig()
	
	if runningMode == 0:
		print 'Running From Live Cam'
		# Program run from live camera
		# load machine learning model from file
		ballMLModel = joblib.load(ballMLFilename)
		goalMLModel = joblib.load(goalMLFilename)
	elif runningMode == 1:
		# Program test mlModel from image
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
	
	while(True):
		# Ini nanti diganti dengan load dari file
		# Create trackbar		
		if runningMode == 0 or runningMode == 4:
			# ini nanti diganti dari live cam
			rawImage = "D:/RoboCupDataset/normal/gambar_normal_" + str(imageNumber) + ".jpg"
		if runningMode == 1 or runningMode == 2 or runningMode == 3:
			rawImage = "D:/RoboCupDataset/normal/gambar_normal_" + str(imageNumber) + ".jpg"
			# rawImage = "D:/RoboCupDataset/dataset_lighting/my_photo-" + str(imageNumber) + ".jpg"
		# print (fileGambar)
		rgbImage = cv2.imread(rawImage)

		# rgbGamma = gammaCorrection(rgbImage, gamma=2.5)

		# ini gak bagus harusnya deklarasi diatas
		fieldMask = np.zeros(rgbImage.shape[:2], np.uint8)
		notFieldMask = 255 * np.ones(rgbImage.shape[:2], np.uint8)
		
		# Color Conversion
		modRgbImage = rgbImage.copy()
		blurRgbImage = cv2.GaussianBlur(rgbImage,(5,5),0)
		grayscaleImage = cv2.cvtColor(blurRgbImage, cv2.COLOR_BGR2GRAY)
		hsvBlurImage = cv2.cvtColor(blurRgbImage, cv2.COLOR_BGR2HSV)
		hsvImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2HSV)

		# Field Green Color Filtering
		fieldGrBinary = cv2.inRange(hsvBlurImage, lowerFieldGr, upperFieldGr)
		fieldGrBinaryErode = cv2.erode(fieldGrBinary, kernel, iterations = edFieldGr[0])
 		fieldGrFinal = cv2.dilate(fieldGrBinaryErode, kernel, iterations = edFieldGr[1])

		# Field Contour Detection
		_, listFieldContours, _ = cv2.findContours(fieldGrFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if len(listFieldContours) > 0:
			fieldContours = sorted(listFieldContours, key=cv2.contourArea, reverse=True)[:1]
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
		ballWhBinary = cv2.inRange(hsvImage, lowerBallWh, upperBallWh)
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
		detectedBall[0] = -888
		detectedBall[1] = -888
		detectedBall[2] = -888
		detectedBall[3] = -888
		detectedBall[4] = -888

		for ballDetectionMode in range(0, 2):
			if ballDetectionMode == 0:
				_, listBallContours, _ = cv2.findContours(ballGrFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			else:
				_, listBallContours, _ = cv2.findContours(ballWhFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			if len(listBallContours) > 0:
				listSortedBallContours = sorted(listBallContours, key=cv2.contourArea, reverse=True)[:5]
				ballContourLen[ballDetectionMode] = len(listSortedBallContours)
				ballContourLen[2] = ballContourLen[0] + ballContourLen[1]
				for ballContour in listSortedBallContours:
					ballTopLeftX, ballTopLeftY, ballWidth, ballHeight = cv2.boundingRect(ballContour)
					if runningMode == 0 or runningMode == 1:
						# Load model from file and run the algorithm with the model
						# Get contour properties
						# Machine learning parameter
						ballMode = ballDetectionMode
						# Aspect Ratio is the ratio of width to height of bounding rect of the object.
						ballAspectRatio = float(ballWidth) / float(ballHeight)
						# Extent is the ratio of contour area to bounding rectangle area.
						ballArea = float(cv2.contourArea(ballContour))
						ballRectArea = float(ballWidth) * float(ballHeight)
						ballExtent = float(ballArea) / float(ballRectArea)
						# Solidity is the ratio of contour area to its convex hull area.
						ballHull = cv2.convexHull(ballContour)
						ballHullArea = cv2.contourArea(ballHull)
						if ballHullArea > 0:
							ballSolidity = float(ballArea) / float(ballHullArea)
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
						ballPrediction = ballMLModel.predict_proba(ballProperties)
						# print ballPrediction
						# Yes, it is a ball
						if ballPrediction[0,1] == 1:
							# Set variable to skip next step							
							cv2.rectangle(modRgbImage, (ballTopLeftX, ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), ballColor, 2)
							detectedBall[0] = ballTopLeftX + ballWidth / 2 # Centre X
							detectedBall[1] = ballTopLeftY + ballHeight / 2 # Centre Y
							detectedBall[2] = ballWidth
							detectedBall[3] = ballHeight
							ballDistance = 0
							detectedBall[4] = ballDistance

							ballFound = True
							break	
					elif runningMode == 2:
						# print ballIteration
						# print ballNumber
						if ballNumber == ballIteration:
							# Machine learning parameter
							ballMode = ballDetectionMode
							# Aspect Ratio is the ratio of width to height of bounding rect of the object.
							ballAspectRatio = float(ballWidth) / float(ballHeight)
							# Extent is the ratio of contour area to bounding rectangle area.
							ballArea = float(cv2.contourArea(ballContour))
							ballRectArea = float(ballWidth) * float(ballHeight)
							ballExtent = float(ballArea) / float(ballRectArea)
							# Solidity is the ratio of contour area to its convex hull area.
							ballHull = cv2.convexHull(ballContour)
							ballHullArea = cv2.contourArea(ballHull)

							if ballHullArea > 0:
								ballSolidity = float(ballArea) / float(ballHullArea)
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
							cv2.rectangle(modRgbImage, (ballTopLeftX,ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), ballColor, 2)
						else:
							# print ballContourLen
							if ballDetectionMode == 0 and ballNumber < ballContourLen[0]:
								cv2.rectangle(modRgbImage, (ballTopLeftX,ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), contourColor, 2)
							elif ballDetectionMode == 1 and ballNumber >= ballContourLen[0]:
								cv2.rectangle(modRgbImage, (ballTopLeftX,ballTopLeftY), (ballTopLeftX + ballWidth, ballTopLeftY + ballHeight), contourColor, 2)
						ballIteration += 1
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
		detectedGoal[0] = -888
		detectedGoal[1] = -888
		detectedGoal[2] = -888
		detectedGoal[3] = -888
		detectedGoal[4] = -888
		detectedGoal[5] = -888
		detectedGoal[6] = -888

		# Field Contour Detection
		_, listGoalContours, _ = cv2.findContours(goalWhFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if len(listGoalContours) > 0:
			listSortedGoalContours = sorted(listGoalContours, key=cv2.contourArea, reverse=True)[:5]
			goalContourLen += len(listSortedGoalContours)
			for goalContour in listSortedGoalContours:
				goalTopLeftX, goalTopLeftY, goalWidth, goalHeight = cv2.boundingRect(goalContour)
				if runningMode == 0 or runningMode == 1:
					goalAspectRatio = float(goalWidth) / float(goalHeight)
					goalArea = float(cv2.contourArea(goalContour))
					goalRectArea = float(goalWidth) * float(goalHeight)
					goalExtent = float(goalArea) / float(goalRectArea)
					goalHull = cv2.convexHull(goalContour)
					goalHullArea = cv2.contourArea(goalHull)
					if goalHullArea > 0:
						goalSolidity = float(goalArea) / float(goalHullArea)
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
					goalPrediction = goalMLModel.predict_proba(goalProperties)

					if goalPrediction[0,1] == 1:
						cv2.rectangle(modRgbImage, (goalTopLeftX,goalTopLeftY), (goalTopLeftX + goalWidth, goalTopLeftY + goalHeight), goalColor, 2)
						# Ini nanti dijadikan array aja
						poleLeftBottom = tuple(goalContour[goalContour[:, :, 0].argmin()] [0] )
						poleRightUp = tuple(goalContour[goalContour[:, :, 0].argmax()] [0])
						poleLeftUp = tuple(goalContour[goalContour[:, :, 1].argmin()][0])
						poleRightBottom = tuple(goalContour[goalContour[:, :, 1].argmax()][0])

						showPole = True
						if showPole == True:
							cv2.circle(modRgbImage, poleLeftBottom, 8, (0, 0, 255), -1)
							cv2.circle(modRgbImage, poleRightUp, 8, (0, 255, 0), -1)
							cv2.circle(modRgbImage, poleLeftUp, 8, (255, 0, 0), -1)
							cv2.circle(modRgbImage, poleRightBottom, 8, (255, 255, 0), -1)

						poleLeftHeight = np.linalg.norm(np.array(poleLeftUp)-np.array(poleLeftBottom))
						poleRightHeight = np.linalg.norm(np.array(poleRightUp)-np.array(poleRightBottom))

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
							goalPoleMoment = cv2.moments(goalPoleContour[0])
							poleMomentPosition[goalPolePosition, 0] = int(goalPoleMoment["m10"] / goalPoleMoment["m00"]) #x
							poleMomentPosition[goalPolePosition, 1] = int(goalPoleMoment["m01"] / goalPoleMoment["m00"]) #y

						# Gambar titik moment
						showMoment = True
						if showMoment == True:
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

						detectedGoal[0] = goalTopLeftX + goalWidth / 2 # X
						detectedGoal[1] = goalTopLeftY + goalHeight / 2
						detectedGoal[2] = poleLeftHeight
						detectedGoal[3] = poleRightHeight
						detectedGoal[4] = poleClass
						poleLeftDistance = 0
						poleRightDistance = 0
						detectedGoal[5] = poleLeftDistance
						detectedGoal[6] = poleRightDistance
						# Udah ketemu ya break aja
						break

				elif runningMode == 3:
					if goalNumber == goalIteration:
						goalAspectRatio = float(goalWidth) / float(goalHeight)
						goalArea = float(cv2.contourArea(goalContour))
						goalRectArea = float(goalWidth) * float(goalHeight)
						goalExtent = float(goalArea) / float(goalRectArea)
						goalHull = cv2.convexHull(goalContour)
						goalHullArea = cv2.contourArea(goalHull)
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
						cv2.rectangle(modRgbImage, (goalTopLeftX,goalTopLeftY), (goalTopLeftX + goalWidth, goalTopLeftY + goalHeight), goalColor, 2)
					else:
						cv2.rectangle(modRgbImage, (goalTopLeftX,goalTopLeftY), (goalTopLeftX + goalWidth, goalTopLeftY + goalHeight), contourColor, 2)
					goalIteration += 1

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
		if runningMode == 0 or runningMode == 1:
			textLine = "Running ==> Image : {} Ball : {} Dataset : {}".format(imageNumber, ballNumber, ballDataNumber)
			cv2.putText(modRgbImage, textLine, (10,20), font, 0.4, (0,0,255), 1, cv2.LINE_AA)
		elif runningMode == 2:
			textLine = "Train Ball ==> Image : {} Ball : {} Dataset : {}".format(imageNumber, ballNumber, ballDataNumber)
			cv2.putText(modRgbImage, textLine, (10,20), font, 0.4, (0,0,255), 1, cv2.LINE_AA)
		elif runningMode == 3:
			textLine = "Train Goal ==> Image : {} Goal : {} Dataset : {}".format(imageNumber, goalNumber, goalDataNumber)
			cv2.putText(modRgbImage, textLine, (10,20), font, 0.4, (0,0,255), 1, cv2.LINE_AA)

		if imageToDisplay == 1:
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
			cv2.imshow("Barelang Vision", modRgbImage)
			cv2.imshow("Ball Green Binary Image", ballGrBinary)
			cv2.imshow("Ball Green Final Image", ballGrFinal)
		elif imageToDisplay == 3: #bola mode 2
			lowerBallWh[0] = cv2.getTrackbarPos('HMin','Control')
			lowerBallWh[1] = cv2.getTrackbarPos('SMin','Control')
			lowerBallWh[2] = cv2.getTrackbarPos('VMin','Control')
			upperBallWh[0] = cv2.getTrackbarPos('HMax','Control')
			upperBallWh[1] = cv2.getTrackbarPos('SMax','Control')
			upperBallWh[2] = cv2.getTrackbarPos('VMax','Control')
			edBallWh[0] = cv2.getTrackbarPos('Erode','Control')
			edBallWh[1] = cv2.getTrackbarPos('Dilate','Control')
			cv2.imshow("Barelang Vision", modRgbImage)
			cv2.imshow("Ball White Binary Image", ballWhFinal)
		elif imageToDisplay == 4:
			lowerGoalWh[0] = cv2.getTrackbarPos('HMin','Control')
			lowerGoalWh[1] = cv2.getTrackbarPos('SMin','Control')
			lowerGoalWh[2] = cv2.getTrackbarPos('VMin','Control')
			upperGoalWh[0] = cv2.getTrackbarPos('HMax','Control')
			upperGoalWh[1] = cv2.getTrackbarPos('SMax','Control')
			upperGoalWh[2] = cv2.getTrackbarPos('VMax','Control')
			edGoalWh[0] = cv2.getTrackbarPos('Erode','Control')
			edGoalWh[1] = cv2.getTrackbarPos('Dilate','Control')
			cv2.imshow("Barelang Vision", modRgbImage)
			cv2.imshow("Goal Binary Image", goalWhFinal)
		else:
			cv2.imshow("Barelang Vision", modRgbImage)
		
		'''
		print '----------BarelangFC-Vision---------------------------------'
		print '### All Running Mode ### -----------------------------------'	
		print 'Parse Field -------------------------------------------- [1]'	
		print 'Parse Ball Green (Mode 1) ------------------------------ [2]'
		print 'Parse Ball White (Mode 2) ------------------------------ [3]'		
		print 'Parse Goal --------------------------------------------- [4]'
		print 'Save Filter Config ------------------------------------- [S]'
		print 'Load Filter Config ------------------------------------- [L]'
		print 'Destroy All Windows ------------------------------------ [0]'
		print 'Help --------------------------------------------------- [H]'
		print 'Exit BarelangFC-Vision --------------------------------- [X]'
		print '### Testing Mode ### ---------------------------------------'
		print 'Next Image --------------------------------------------- [N]'
		print 'Previous Image ----------------------------------------- [P]'
		print '### Training Mode ### --------------------------------------'
		print 'Next Contour ------------------------------------------- [C]'
		print 'Previous Contour --------------------------------------- [Z]'
		print '### Ball Training Mode ### ---------------------------------'
		print 'Mark as Ball ------------------------------------------- [B]'
		print 'Mark as Not Ball --------------------------------------- [U]'
		print 'Train Ball Dataset and Save ML Model ------------------- [T]'
		print 'Save Ball Dataset to CSV ------------------------------- [D]'
		print 'Load Ball Dataset from CSV, Train and Save ML Model ---- [M]' 
		print '### Goal Training Mode ### ---------------------------------'
		print 'Mark as Goal ------------------------------------------- [G]'
		print 'Mark as Not Goal --------------------------------------- [U]'
		print 'Train Goal Dataset and Save ML Model ------------------- [T]'
		print 'Save Goal Dataset to CSV ------------------------------- [D]'
		print 'Load Goal Dataset from CSV, Train and Save ML Model ---- [M]' 
		'''
		# Waiting keyboard interrupt
		k = cv2.waitKey(1)
		# Keyboard shortcut for running mode
		if runningMode == 0:
			if k == ord('x'):
				imageToDisplay = 0
				cv2.destroyAllWindows()
				print 'Exit Program'
				break
			elif k == ord('1'):
				cv2.destroyAllWindows()			
				imageToDisplay = 1
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Field Parameter'
			elif k == ord('2'): 
				cv2.destroyAllWindows()
				imageToDisplay = 2 
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball Green Parameter'
			elif k == ord('3'): 
				cv2.destroyAllWindows()
				imageToDisplay = 3 
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball White Parameter'
			elif k == ord('4'):
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
			elif k == ord('0'):
				imageToDisplay = 0
				cv2.destroyAllWindows()
				print 'Close All Windows'
		# Keyboard Shortcut for Dataset Testing From Image
		elif runningMode == 1:
			# print 'asdasd'
			if k == ord('x'):
				imageToDisplay = 0
				cv2.destroyAllWindows()
				print 'Exit Program'
				break
			elif k == ord('1'):
				cv2.destroyAllWindows()			
				imageToDisplay = 1
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Field Parameter'
			elif k == ord('2'): 
				cv2.destroyAllWindows()
				imageToDisplay = 2 
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball Green Parameter'
			elif k == ord('3'): 
				cv2.destroyAllWindows()
				imageToDisplay = 3 
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball White Parameter'
			elif k == ord('4'):
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
			elif k == ord('0'):
				imageToDisplay = 0
				cv2.destroyAllWindows()
				print 'Close All Windows'
			elif k == ord('n'):
				imageNumber += 1
				print 'Next Image'
			elif k == ord('p'):
				imageNumber -= 1
				print 'Previous Image'
		# Keyboard shortcut for ball training mode
		elif runningMode == 2:
			if k == ord('x'):
				imageToDisplay = 0
				cv2.destroyAllWindows()
				np.savetxt(ballDatasetFilename, npBallDataset, fmt='%.5f', delimiter=',', header="Samples,  Aspect Ratio,  Area,  Rect Area, Extent,  Solidity,  H0,  H1, H2, H3, H4, Mode, Ball")
				print 'Exit Program'
				break
			elif k == ord('1'):
				cv2.destroyAllWindows()			
				imageToDisplay = 1
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Field Parameter'
			elif k == ord('2'): 
				cv2.destroyAllWindows()
				imageToDisplay = 2 
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball Green Parameter'
			elif k == ord('3'): 
				cv2.destroyAllWindows()
				imageToDisplay = 3 
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Ball White Parameter'
			elif k == ord('s'):
				saveConfig()
				print 'Save Setting Value'
			elif k == ord('l'):
				loadConfig()
				loadTrackbars(imageToDisplay)
				print 'Load Setting Value'
			elif k == ord('0'):
				imageToDisplay = 0
				cv2.destroyAllWindows()
				print 'Close All Windows'
			elif k == ord('n'):
				imageNumber += 1
				print 'Next Image'
			elif k == ord('p'):
				imageNumber -= 1
				print 'Previous Image'
			elif k == ord('c'):
				ballNumber += 1
				if ballNumber >= ballContourLen[2]:
					ballNumber = 0
					imageNumber += 1
				print 'Next Ball Contour'
			elif k == ord('z'):
				ballNumber -= 1
				if ballNumber < 0:
					ballNumber = 0
					imageNumber -= 1
				print 'Previous Ball Contour'
			elif k == ord('b'):
				isBall = 1
				npBallData = np.array([ballDataNumber, ballAspectRatio, ballArea, ballRectArea, ballExtent, ballSolidity, ballHistogram0[0], ballHistogram1[0], ballHistogram2[0], ballHistogram3[0], ballHistogram4[0], ballMode, isBall])
				npBallDataset = np.insert(npBallDataset, ballDataNumber-1, npBallData, axis=0)
				ballNumber += 1
				if ballNumber >= ballContourLen[2]:
					ballNumber = 0
					imageNumber += 1
				ballDataNumber += 1
				print 'Mark as Ball'
			elif k == ord('u'):
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
				imageToDisplay = 0
				cv2.destroyAllWindows()
				np.savetxt(goalDatasetFilename, npGoalDataset, fmt='%.5f', delimiter=',', header="Samples,  Aspect Ratio,  Area,  Rect Area, Extent,  Solidity,  H0,  H1, H2, H3, H4, Goal")
				print 'Exit Program'
				break
			elif k == ord('1'):
				cv2.destroyAllWindows()			
				imageToDisplay = 1
				createTrackbars(imageToDisplay)
				loadTrackbars(imageToDisplay)
				print 'Setting Field Parameter'
			elif k == ord('4'):
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
			elif k == ord('0'):
				imageToDisplay = 0
				cv2.destroyAllWindows()
				print 'Close All Windows'
			elif k == ord('n'):
				imageNumber += 1
				print 'Next Image'
			elif k == ord('p'):
				imageNumber -= 1
				print 'Previous Image'
			elif k == ord('c'):
				goalNumber += 1
				if goalNumber >= goalContourLen:
					goalNumber = 0
					imageNumber += 1
				print 'Next Goal Contour'
			elif k == ord('z'):
				goalNumber -= 1
				if goalNumber < 0:
					goalNumber = 0
					imageNumber -= 1
				print 'Previous Goal Contour'
			elif k == ord('g'):
				isGoal = 1
				npGoalData = np.array([goalDataNumber, goalAspectRatio, goalArea, goalRectArea, goalExtent, goalSolidity, goalHistogram0[0], goalHistogram1[0], goalHistogram2[0], goalHistogram3[0], goalHistogram4[0], isGoal])
				npGoalDataset = np.insert(npGoalDataset, goalDataNumber-1, npGoalData, axis=0)
				goalNumber += 1
				if goalNumber >= goalContourLen:
					goalNumber = 0
					imageNumber += 1
				goalDataNumber += 1
				print 'Mark as Goal'
			elif k == ord('u'):
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

'''
def main_lama():
	#######################
	# Create a black image, a window
	cv2.namedWindow('Barelang Vision ')
	cap = cv2.VideoCapture(0)
	#cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
	#cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);

	help()
	loadConfig()
	while(1):
		#time_start = datetime.datetime.now()
		#video query from webcam
		ret, im = cap.read()

		# Read image
		#im = cv2.imread("images/acak%d.jpg"%imageNumber)
		#im = cv2.imread("/home/barelangfc/Foto_BolaPutih/my_photo-%d.jpg"%imageNumber)
		#im = cv2.imread("/home/barelangfc/python/my_photo-1.jpg")
		#print("images/acak%d.jpg"%imageNumber)

		image = cv2.resize(im, (im_width, im_height),interpolation = cv2.INTER_AREA)
		grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
		#hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)

		#cv2.imshow("Color Conversion", hsv_image)

		#detection field
		#blur_image = cv2.blur(image, (fblur,fblur))
		hsv_blur_image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB) #field hijau
		#cv2.imshow("Field Blur Image", hsv_blur_image)
		f_lower_val = np.array([hfmin,sfmin,vfmin])
		f_upper_val = np.array([hfmax,sfmax,vfmax])
		parsefield = cv2.inRange(hsv_blur_image,f_lower_val,f_upper_val)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
		dilate_parsefield = cv2.dilate(parsefield, kernel, imageNumbers = dfsize)
		#erode_parsefield = cv2.erode(parsefield,(20,20),imageNumbers = efsize)
		#dilate_parsefield = cv2.dilate(erode_parsefield,(20,20),imageNumbers = dfsize)

		#detection goal
		goal_colorConv_image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
		g_lower_val = np.array([hgmin,sgmin,vgmin])
		g_upper_val = np.array([hgmax,sgmax,vgmax])
		parsegoal = cv2.inRange(goal_colorConv_image,g_lower_val,g_upper_val)
		erode_parsegoal = cv2.erode(parsegoal,kernel,imageNumbers = egsize)
		dilate_parsegoal = cv2.dilate(erode_parsegoal,kernel,imageNumbers = dgsize)

		# bola mode 1 --> threshold hijau
		# Invert warna lapangan untuk deteksi bola
		#n_lower_val = np.array([hnmin,snmin,vnmin])
		#n_upper_val = np.array([hnmax,snmax,vnmax])
		#parseball_inv_binary = cv2.inRange(hsv_image,n_lower_val,n_upper_val)
		#parseball_invert_field = cv2.bitwise_not(parseball_inv_binary)
		#erode_parseball_invert_field = cv2.erode(parseball_invert_field,(10,10),ensize) #5
		#parseball_invert_field = cv2.dilate(parseball_invert_field,(10,10),dnsize)

		# bola mode 2 --> threshold warna putih
		#detection goal

		#parseball_white = cv2.dilate(parseball_white,(5,5),dbsize)

		# checkObject
		#debug_mode = cv2.getTrackbarPos('Object','Control')

		###################
		# Detecttion Field --> Rectangle Green, source = blur image
		_, f_contours, _ = cv2.findContours(dilate_parsefield.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if len(f_contours) > 0:
			f_cntr = max(f_contours, key=cv2.contourArea)
			hull = cv2.convexHull(f_cntr)
			mask = np.zeros(image.shape[:2], np.uint8)

			#mask = np.zeros([im_height, im_width, 3], dtype=np.uint8)
			#mask[100:300, 100:400] = 255
			cv2.drawContours(mask, [hull], 0, 255,cv2.FILLED)
			field_image = cv2.bitwise_and(image,image,mask = mask)
			cv2.drawContours(image, [hull], 0, (0,255,0),2)
		else:
			field_image = xxxImage
		#cv2.drawContours(image, f_contours, -1, (0,255,0),5)
		hsv_field_image = cv2.cvtColor(field_image,cv2.COLOR_BGR2HSV)
		# bola mode 1 --> threshold hijau
		# Invert warna lapangan untuk deteksi bola
		n_lower_val = np.array([hnmin,snmin,vnmin])
		n_upper_val = np.array([hnmax,snmax,vnmax])
		parseball_inv_binary = cv2.inRange(hsv_field_image,n_lower_val,n_upper_val)
		parseball_invert_field = cv2.bitwise_not(parseball_inv_binary)
		erode_parseball_invert_field = cv2.erode(parseball_invert_field,kernel,imageNumbers = ensize) #5
		dilate_parseball_invert_field = cv2.dilate(erode_parseball_invert_field,kernel,imageNumbers = dnsize)
		#print '%d'%(ensize)

		ball_colorConv_image = cv2.cvtColor(field_image,cv2.COLOR_BGR2YUV)
		b_lower_val = np.array([hbmin,sbmin,vbmin])
		b_upper_val = np.array([hbmax,sbmax,vbmax])
		parseball_white = cv2.inRange(ball_colorConv_image,b_lower_val,b_upper_val)
		erode_parseball_white = cv2.erode(parseball_white,kernel,imageNumbers = ebsize) #2
		dilate_parseball_white = cv2.dilate(erode_parseball_white,kernel,imageNumbers = dbsize) #2
		#cv2.imshow("erode", erode_parseball_white)

		#cv2.rectangle(image_scaled, (field_x,field_y), (field_x + field_w, field_y + field_h), (0, 255, 0), 3)
		#cv2.rectangle(image, (field_xtop,field_ytop), (field_xbot, field_ybot), (0, 255, 0), 3)
		#f_rect = cv2.minAreaRect(f_cntr)
		#f_box = cv2.boxPoints(f_rect)
		#f_box = np.int0(f_box)
		#cv2.drawContours(image,[f_box],0,(0,255,0),5)

		###################
		# Ball detection mode 1 --> dot Circle orange
		#field_roi = erode_parseball_invert_field[field_ytop:field_ybot, field_xtop:field_xbot]
		_, b_contours, _ = cv2.findContours(dilate_parseball_invert_field.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		#field_roi_wh = erode_parseball_white[field_ytop:field_ybot, field_xtop:field_xbot]
		_, b_wh_contours, _ = cv2.findContours(dilate_parseball_white.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		center_bola = im_width/2,im_height/2

		#ball_area_param = cv2.getTrackbarPos('B Area','Control')
		#ball_extent_param = cv2.getTrackbarPos('B Area_Ratio','Control')
		#ball_extent_param = float(ball_extent_param) / 10
		#cv2.drawContours(image, b_contours, -1, (0,255,255),3)
		#if field_w !=0 and field_h != 0:
		ball_found =  False
		if (ball_found == False and mod1 == False and mod2 == False) or (mod1 == True and mod2 == False) :
			if len(b_contours) > 0:
				b_sorted_contours = sorted(b_contours, key=cv2.contourArea, reverse=True)[:5]
				ball_found =  False
				ball_imageNumber = 1;
				#cv2.drawContours(image, b_sorted_contours, -1, (0,255,255),3)
				for b_cntr in b_sorted_contours:
					# Initialize ball parameter value
					ball_centre_x = -1
					ball_centre_y = -1
					ball_width = 0
					ball_height = 0
					ball_area = 0
					ball_rect_area = 0
					ball_area_ratio = 0
					ball_wh_ratio = 0
					percent_white = 0
					ball_radius = 0

					#cv2.drawContours(image,b_cntr,-1,(0,255,255),5)
					ball_topleft_x, ball_topleft_y, ball_width, ball_height = cv2.boundingRect(b_cntr)
					# koreksi dengan koordinat image_scaled_
					ball_topleft_x = ball_topleft_x
					ball_topleft_y = ball_topleft_y
					ball_centre_x = ball_topleft_x + (ball_width/2)
					ball_centre_y = ball_topleft_y + (ball_height/2)
					ball_botleft_x = ball_topleft_x
					ball_botleft_y = ball_topleft_y + ball_height
					ball_topright_x = ball_topleft_x + ball_width
					ball_topright_y = ball_topleft_y
					ball_botright_x = ball_topleft_x + ball_width
					ball_botright_y = ball_topleft_y + ball_height
					ball_radius = ball_width/2

					#cv2.drawContours(image,[box],0,(0,255,255),2)
					ball_area = (float(cv2.contourArea(b_cntr)) / float(im_area)) * 100.0
					ball_rect_area = (float (ball_width * ball_height) / float(im_area)) * 100.0
					# Handle exception devide by zero
					if ball_rect_area != 0:
						ball_area_ratio = float(ball_area) / float(ball_rect_area)
					if ball_height != 0:
						ball_wh_ratio = float(ball_width) / float(ball_height)
					predicted_ball = grayscale_image[ball_topleft_y:ball_topleft_y + ball_height, ball_topleft_x:ball_topleft_x + ball_width]
					hist = cv2.calcHist([predicted_ball], [0], None, [5], [0, 256])
					hist_val_0, hist_val_1, hist_val_2, hist_val_3, hist_val_4 = hist
					sum_hist = hist_val_0 + hist_val_1 + hist_val_2 + hist_val_3 + hist_val_4
					#cv2.rectangle(image, (ball_topleft_x,ball_topleft_y), (ball_topleft_x + ball_width, ball_topleft_y+ball_height), (0,0,255), 3)

					if sum_hist > 0:
						percent_white = (float(hist_val_4) / float(sum_hist)) * 100.0

					if debug_ballmode1 == 1:
						selected_ball = cv2.getTrackbarPos('Ball','Control')
						if selected_ball == ball_imageNumber:
							print 'Ball Number %d ==> X = %d Y = %d W = %d H = %d Radius = %d Area = %.2f R_Area = %.2f Area_Rat = %.2f WH_Rat = %.2f Percent_Wh = %.2f'%(ball_imageNumber,ball_centre_x,ball_centre_y,ball_width,ball_height,ball_radius,ball_area,ball_rect_area,ball_area_ratio,ball_wh_ratio, percent_white)
							ball_color = (244, 66, 66)
						else:
							ball_color = (31,127,255)
						cv2.rectangle(image, (ball_topleft_x,ball_topleft_y), (ball_topleft_x + ball_width, ball_topleft_y+ball_height), ball_color, 3)
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
					ball_imageNumber += 1;

			# Ball detection mode 2
			# mode 2 digunakan khusus bola dekat yg tdk bisa terdeteksi oleh mode 1
			# tambah deteksi lokasi center terhadap warna hitam lapangan
			#if field_w !=0 and field_h != 0:
		if (ball_found == False and mod2 == False and mod1 == False) or (mod2 == True and mod1 == False) :
			if len(b_wh_contours) > 0:
				b_wh_sorted_contours = sorted(b_wh_contours, key=cv2.contourArea, reverse=True)[:5]
				ball_found =  False
				ball_imageNumber = 1;
				for b_wh_cntr in b_wh_sorted_contours:
					# Initialize ball parameter value
					ball_centre_x = -1
					ball_centre_y = -1
					ball_width = 0
					ball_height = 0
					ball_area = 0
					ball_rect_area = 0
					ball_area_ratio = 0
					ball_wh_ratio = 0
					percent_white = 0
					ball_radius = 0

					# pending dulu
					#ball_wh_rect = cv2.minAreaRect(b_wh_cntr)
					#ball_wh_box = cv2.boxPoints(ball_wh_rect)
					#ball_wh_box = np.int0(ball_wh_box)
					# offset bola
					#ball_rot_rect = perspective.order_points(ball_wh_box)

					#cv2.drawContours(image,b_cntr,-1,(0,255,255),5)
					ball_topleft_x, ball_topleft_y, ball_width, ball_height = cv2.boundingRect(b_wh_cntr)
					# koreksi dengan koordinat image_scaled_
					ball_topleft_x = ball_topleft_x
					ball_topleft_y = ball_topleft_y
					ball_botleft_x = ball_topleft_x
					ball_botleft_y = ball_topleft_y + ball_height
					ball_topright_x = ball_topleft_x + ball_width
					ball_topright_y = ball_topleft_y
					ball_botright_x = ball_topleft_x + ball_width
					ball_botright_y = ball_topleft_y + ball_height
					ball_radius = ball_width/2

					ball_centre_x = ball_topleft_x + (ball_width/2)
					ball_centre_y = ball_topleft_y + (ball_height/2)
					#cv2.drawContours(image,[box],0,(0,255,255),2)
					ball_area = (float(cv2.contourArea(b_wh_cntr)) / float(im_area)) * 100.0
					ball_rect_area = (float (ball_width * ball_height) / float(im_area)) * 100.0
					# Handle exception devide by zero
					if ball_rect_area != 0:
						ball_area_ratio = float(ball_area) / float(ball_rect_area)
					if ball_height != 0:
						ball_wh_ratio = float(ball_width) / float(ball_height)

					predicted_ball = grayscale_image[ball_topleft_y:ball_topleft_y + ball_height, ball_topleft_x:ball_topleft_x + ball_width]
					hist = cv2.calcHist([predicted_ball], [0], None, [5], [0, 256])
					hist_val_0, hist_val_1, hist_val_2, hist_val_3, hist_val_4 = hist
					sum_hist = hist_val_0 + hist_val_1 + hist_val_2 + hist_val_3 + hist_val_4
					# exception handling
					if sum_hist > 0:
						percent_white = (float(hist_val_4) / float(sum_hist)) * 100.0
					# Hitung pixel white di titik persegi
					#print '%d %d'%(ball_topleft_x,ball_topleft_y)
					#print '%d %d'%(ball_botleft_x,ball_botleft_y)
					#print '%d %d'%(ball_topright_x,ball_topright_y)
					#print '%d %d'%(ball_botright_x,ball_botright_y)
					#cv2.rectangle(image, (ball_topleft_x,ball_topleft_y), (ball_topleft_x + ball_width, ball_topleft_y + ball_height), (255, 255, 255), 2) #(244,66,66)

					if debug_ballmode2 == 1:
						selected_ball = cv2.getTrackbarPos('Ball','Control')
						if selected_ball == ball_imageNumber:
							print 'Ball Number %d ==> X = %d Y = %d W = %d H = %d Radius = %d Area = %.2f R_Area = %.2f Area_Rat = %.2f WH_Rat = %.2f Percent_Wh = %.2f'%(ball_imageNumber,ball_centre_x,ball_centre_y,ball_width,ball_height,ball_radius,ball_area,ball_rect_area,ball_area_ratio,ball_wh_ratio, percent_white)
							ball_color = (244, 66, 66)
						else:
							ball_color = (31,127,255)
							#cv2.drawContours(image,[ball_wh_box],0,ball_color,3)
						cv2.rectangle(image, (ball_topleft_x,ball_topleft_y), (ball_topleft_x + ball_width, ball_topleft_y+ball_height), ball_color, 3) #ball_color
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
					ball_imageNumber += 1;

		if ball_found == False:
			ball_centre_x = -1
			ball_centre_y = -1
			ball_width = 0
			ball_height = 0
			ball_radius = 0
			ball_area = 0
			ball_rect_area = 0
			ball_area_ratio = 0
			ball_wh_ratio = 0
			percent_white = 0
			ball_radius = 0


		_, g_contours, _ = cv2.findContours(dilate_parsegoal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
		goal_found = False
		if len(g_contours) > 0:
			g_sorted_contours = sorted(g_contours, key=cv2.contourArea, reverse=True)[:10]
			goal_imageNumber = 1;
			for g_cntr in g_sorted_contours:
				x_box,y_box,w_box,h_box = cv2.boundingRect(g_cntr)
				goal_centre_x_box = x_box + (w_box // 2)
				goal_centre_y_box = y_box + (h_box // 2)

				goal_rect = cv2.minAreaRect(g_cntr)
				goal_box = cv2.boxPoints(goal_rect)
				goal_box = np.int0(goal_box)
				goal_rot_rect = perspective.order_points(goal_box)

				# the midpoint between bottom-left and bottom-right coordinates
				(goal_top_left, goal_top_right, goal_bottom_right, goal_bottom_left) = goal_box
				#ft, goal_top_left, goal_bottom_right, goal_top_right) = goal_box
				(goal_midtop_x, goal_midtop_y) = midpoint(goal_top_left, goal_top_right)
				(goal_midbot_x, goal_midbot_y) = midpoint(goal_bottom_left, goal_bottom_right)
				(goal_midleft_x, goal_midleft_y) = midpoint(goal_top_left, goal_bottom_left)
				(goal_midright_x, goal_midright_y) = midpoint(goal_top_right, goal_bottom_right)
				#(goal_centre_x, goal_centre_y) = midpoint((goal_midtop_x, goal_midtop_y), (goal_midbot_x, goal_midbot_y))
				goal_centre_x = goal_centre_x_box
				goal_centre_y = goal_centre_y_box

				goal_height = dist.euclidean((goal_midtop_x, goal_midtop_y), (goal_midbot_x, goal_midbot_y))
				goal_width = dist.euclidean((goal_midleft_x, goal_midleft_y), (goal_midright_x, goal_midright_y))
				goal_area = (float(cv2.contourArea(g_cntr)) / float(im_area))* 100.0
				goal_rect_area = (float(goal_height * goal_width) / float(im_area)) * 100.0  # Calculate ball_area of rectangle
				# Handle error div by zero
				if goal_rect_area != 0:
					goal_area_ratio = float(goal_area) / float(goal_rect_area)
				if goal_height != 0:
					goal_wh_ratio = float(goal_width) / float(goal_height)
				#
				if debug_goal == 1:
					selected_goal = cv2.getTrackbarPos('Goal','Control')
					if selected_goal == goal_imageNumber :
						print 'Goal Num %d ==> X = %d Y = %d W = %d H = %d Area = %.2f R_Area = %.2f Area_Rat = %.2f WH_Rat = %.2f'%(goal_imageNumber, goal_centre_x,goal_centre_y,goal_width,goal_height,goal_area,goal_rect_area,goal_area_ratio,goal_wh_ratio)
						goal_color = (0, 0, 255)
					else:
						goal_color = (255, 255, 255)
					cv2.drawContours(image,[goal_box],0,goal_color,3)
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
				goal_imageNumber += 1;
		if goal_found == False:
			goal_centre_x = -1
			goal_centre_y = -1
			goal_width = 0
			goal_height = 0
			goal_area = 0
			goal_rect_area = 0
			goal_area_ratio = 0
			goal_wh_ratio = 0

		###################################
		## sendSocket LocalHost UDP
		try:
			#s.flush()
			msg = '%d,%d,%d,%d'%(ball_centre_x, ball_centre_y, goal_centre_x, goal_centre_y)
			s.sendto(msg, (host, port))
		except socket.error:
			#print 'Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
			sys.exit()

		if stream :
			font = cv2.FONT_HERSHEY_SIMPLEX
			textLine1 = 'Ball ==> X = %d Y = %d D = %d'%(ball_centre_x, ball_centre_y, 0)
			textLine2 = 'Goal ==> X = %d Y = %d D = %d'%(goal_centre_x, goal_centre_y, 0)
			cv2.putText(image,textLine1,(10,20), font, 0.4,(0,0,255),1,cv2.LINE_AA)
			cv2.putText(image,textLine2,(10,35), font, 0.4,(0,0,255),1,cv2.LINE_AA)
			cv2.imshow("Barelang Vision ", image)

			#display_image = cv2.getTrackbarPos('Image','Control')
			#cv2.imshow("Control", dummyImage)
			#print display_image
			if display_image == 1:
				hfmax = cv2.getTrackbarPos('HMax','Control')
				hfmin = cv2.getTrackbarPos('HMin','Control')
				sfmax = cv2.getTrackbarPos('SMax','Control')
				sfmin = cv2.getTrackbarPos('SMin','Control')
				vfmax = cv2.getTrackbarPos('VMax','Control')
				vfmin = cv2.getTrackbarPos('VMin','Control')
				efsize = cv2.getTrackbarPos('Erode','Control')
				dfsize = cv2.getTrackbarPos('Dilate','Control')
				#fblur = cv2.getTrackbarPos('Field Blur','Control') * 10
				if fblur < 1:
					fblur=1
				#fblur = fblur * 10
				#print fblur
				cv2.imshow("Parse Field", dilate_parsefield)
				cv2.imshow("Field Image", field_image)
				#cv2.imshow("Blur", blur_image)
			elif display_image == 2:
				hgmax = cv2.getTrackbarPos('HMax','Control')
				hgmin = cv2.getTrackbarPos('HMin','Control')
				sgmax = cv2.getTrackbarPos('SMax','Control')
				sgmin = cv2.getTrackbarPos('SMin','Control')
				vgmax = cv2.getTrackbarPos('VMax','Control')
				vgmin = cv2.getTrackbarPos('VMin','Control')
				egsize = cv2.getTrackbarPos('Erode','Control')
				dgsize = cv2.getTrackbarPos('Dilate','Control')
				debug_goal = cv2.getTrackbarPos('Debug Goal','Control')
				cv2.imshow("Parse Goal", dilate_parsegoal)
			elif display_image == 3: #bola mode 2
				hbmax = cv2.getTrackbarPos('HMax','Control')
				hbmin = cv2.getTrackbarPos('HMin','Control')
				sbmax = cv2.getTrackbarPos('SMax','Control')
				sbmin = cv2.getTrackbarPos('SMin','Control')
				vbmax = cv2.getTrackbarPos('VMax','Control')
				vbmin = cv2.getTrackbarPos('VMin','Control')
				ebsize = cv2.getTrackbarPos('Erode','Control')
				dbsize = cv2.getTrackbarPos('Dilate','Control')
				#print ebsize
				debug_ballmode2 = cv2.getTrackbarPos('Debug Ball Mode2','Control')
				cv2.imshow("Parse Ball Mode2", dilate_parseball_white)
			elif display_image == 4: #bola mode 1
				hnmax = cv2.getTrackbarPos('HMax','Control')
				hnmin = cv2.getTrackbarPos('HMin','Control')
				snmax = cv2.getTrackbarPos('SMax','Control')
				snmin = cv2.getTrackbarPos('SMin','Control')
				vnmax = cv2.getTrackbarPos('VMax','Control')
				vnmin = cv2.getTrackbarPos('VMin','Control')
				ensize = cv2.getTrackbarPos('Erode','Control')
				dnsize = cv2.getTrackbarPos('Dilate','Control')
				debug_ballmode1 = cv2.getTrackbarPos('Debug Ball Mode1','Control')
				#print ensize
				cv2.imshow("Parse Ball Mode 1", dilate_parseball_invert_field)

		k = cv2.waitKey(1)
		if k == ord('x'):
			break
		elif k == ord('s'):
			saveConfig()
		elif k == ord('l'):
			loadConfig()
			loadTrackbars(display_image)
		elif k == ord('r'):
			display_image = debug_ballmode1 = debug_ballmode2 = debug_goal = 0
			cv2.destroyAllWindows()
			mod1 = mod2 = False
		elif k == ord('d'):
			imageNumber += 1
		elif k == ord('a'):
			imageNumber -= 1

		elif k == ord('f'):
			cv2.destroyAllWindows()
			createTrackbars(1)
			display_image = 1
			loadTrackbars(display_image)
			print'field..'

		elif k == ord('g'):
			cv2.destroyAllWindows()
			createTrackbars(2)
			display_image = 2 #debug_mode = 2
			loadTrackbars(display_image)
			print'goal..'

		elif k == ord('b'): #bola mode 2 parse white
			cv2.destroyAllWindows()
			createTrackbars(3)
			display_image = 3 #debug_mode = 3
			loadTrackbars(display_image)
			mod2 = True
			mod1 = False
			print'ball mode 2 parse white..'

		elif k == ord('n'): #bola mode 1 invert field
			cv2.destroyAllWindows()
			createTrackbars(4)
			display_image = 4 #debug_mode = 4
			loadTrackbars(display_image)
			mod2 = False
			mod1 = True
			print'ball mode 1 invert field..'

		#time_end = datetime.datetime.now()
		#time_elapsed = (time_end - time_start).total_seconds()
		#fps = 1.0 / time_elapsed
		#print 'FPS = %.2f'%fps
		#    print 'Ball Param ==> X = %d Y = %d W = %d H = %d Area = %d R_Area = %d Area_Rat = %.2f WH_Rat = %.2f Percent_Wh = %.2f'%(ball_centre_x,ball_centre_y,ball_width,ball_height,ball_area,ball_rect_area,ball_area_ratio,ball_wh_ratio, percent_white)
		#    print 'Goal Param ==> X = %d Y = %d W = %d H = %d Area = %d R_Area = %d Area_Rat = %.2f WH_Rat = %.2f'%(goal_centre_x,goal_centre_y,goal_width,goal_height,goal_area,goal_rect_area,goal_area_ratio,goal_wh_ratio)

	cv2.destroyAllWindows()
'''
if __name__ == "__main__":
    main()