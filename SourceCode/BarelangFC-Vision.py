# Python 2.7 BarelangFC-Vision.py

#######################
## Standard imports
import os
import cv2
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

#######################
##global variable
dummyImage = np.zeros((120,750,3), np.uint8)
xxxImage = np.zeros((640,480,3), np.uint8)
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

# Initialize goal parameter value
goal_centre_x = -1
goal_centre_y = -1
goal_width = 0
goal_height = 0
goal_area = 0
goal_rect_area = 0
goal_area_ratio = 0
goal_wh_ratio = 0

# Configuration
im_width = 640
im_height = 480
im_area = im_width * im_height
iteration = 1
debug_mode = 1
debug_goal = 0
debug_ballmode1 = 0
debug_ballmode2 = 0
display_image = 0
stream = True
mod1 = False
mod2 = False

#######################
##socket
import socket
import sys
host = 'localhost';
port = 2000;
try:
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error:
	print 'Failed to create socket'
	sys.exit()

#######################
##functions
def help():
	print''
	print'----------BarelangFC-Vision-----------'
	print'Help ----------------------------- [H]'
	print'Parse Ball Not Field (Mode 1)----- [N]'
	print'Parse Ball White-----(Mode 2)----- [B]'
	print'Parse Field ---------------------- [F]'
	print'Parse Goal ----------------------- [G]'
	print'Save Config ---------------------- [S]'
	print'Load Config ---------------------- [L]'
	print'Next Iterations [Image View] ----- [A]'
	print'Previous Iterations [Image View] - [D]'
	print'Destroy All Windows -------------- [R]'
	print'Exit BarelangFC-Vision ----------- [X]'
	print''
	return None

def createTrackbars(mode):
	cv2.namedWindow('Control')
	#field
	if mode==1:
		cv2.createTrackbar('Field Blur','Control',0,10,nothing)
		#cv2.createTrackbar('Dilate','Control',0,10,nothing)
	#goal
	elif mode==2:
		cv2.createTrackbar('Debug Goal','Control',0,1,nothing)
		cv2.createTrackbar('Goal','Control',0,20,nothing)
	#ball
	elif mode==3:
		cv2.createTrackbar('Debug Ball Mode2','Control',0,1,nothing)
		cv2.createTrackbar('Ball','Control',0,20,nothing)

	elif mode==4:
		cv2.createTrackbar('Debug Ball Mode1','Control',0,1,nothing)
		cv2.createTrackbar('Ball','Control',0,20,nothing)

	cv2.createTrackbar('HMax','Control',255,255,nothing)
	cv2.createTrackbar('HMin','Control',0,255,nothing)
	cv2.createTrackbar('SMax','Control',255,255,nothing)
	cv2.createTrackbar('SMin','Control',0,255,nothing)
	cv2.createTrackbar('VMax','Control',255,255,nothing)
	cv2.createTrackbar('VMin','Control',0,255,nothing)

	#if mode !=1:
	cv2.createTrackbar('Erode','Control',0,10,nothing)
	cv2.createTrackbar('Dilate','Control',0,100,nothing)

	return None

def loadTrackbars(mode):
	#field
	if mode==1:
		loadHighH = hfmax
		loadLowH  = hfmin
		loadHighS = sfmax
		loadLowS  = sfmin
		loadHighV = vfmax
		loadLowV  = vfmin
		loadEsize = efsize
		loadDsize = dfsize
	#goal
	elif mode==2:
		loadHighH = hgmax
		loadLowH  = hgmin
		loadHighS = sgmax
		loadLowS  = sgmin
		loadHighV = vgmax
		loadLowV  = vgmin
		loadEsize = egsize
		loadDsize = dgsize
	#ball mode 2 / parseWhite
	elif mode==3:
		loadHighH = hbmax
		loadLowH  = hbmin
		loadHighS = sbmax
		loadLowS  = sbmin
		loadHighV = vbmax
		loadLowV  = vbmin
		loadEsize = ebsize
		loadDsize = dbsize
	#ball mode 1 / notField
	elif mode==4:
		loadHighH = hnmax
		loadLowH  = hnmin
		loadHighS = snmax
		loadLowS  = snmin
		loadHighV = vnmax
		loadLowV  = vnmin
		loadEsize = ensize
		loadDsize = dnsize

	cv2.setTrackbarPos('HMax','Control',loadHighH)
	cv2.setTrackbarPos('HMin','Control',loadLowH)
	cv2.setTrackbarPos('SMax','Control',loadHighS)
	cv2.setTrackbarPos('SMin','Control',loadLowS)
	cv2.setTrackbarPos('VMax','Control',loadHighV)
	cv2.setTrackbarPos('VMin','Control',loadLowV)
	#if mode != 1:
	cv2.setTrackbarPos('Erode','Control',loadEsize)
	cv2.setTrackbarPos('Dilate','Control',loadDsize)
	#else :
	#	cv2.setTrackbarPos('Field Blur','Control',fblur/10)
	return None

def saveConfig():
	f = open("storageThreshold.txt","w")
	data = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d'%(hfmax,hfmin,sfmax,sfmin,vfmax,vfmin,efsize,dfsize,hgmax,hgmin,sgmax,sgmin,vgmax,vgmin,egsize,dgsize,hbmax,hbmin,sbmax,sbmin,vbmax,vbmin,ebsize,dbsize,hnmax,hnmin,snmax,snmin,vnmax,vnmin,ensize,dnsize,fblur)
	f.write(data)
	f.close()
	print 'saved'
	return None

def loadConfig():
	global hfmax, hfmin, sfmax, sfmin, vfmax, vfmin, efsize, dfsize
	global hgmax, hgmin, sgmax, sgmin, vgmax, vgmin, egsize, dgsize
	global hbmax, hbmin, sbmax, sbmin, vbmax, vbmin, ebsize, dbsize
	global hnmax, hnmin, snmax, snmin, vnmax, vnmin, ensize, dnsize
	global fblur

	f = open("storageThreshold.txt","r")
	for line in f.readlines():
		#print line
		arr_read = line.split(',')
		hfmax = int(arr_read[0])
		hfmin = int(arr_read[1])
		sfmax = int(arr_read[2])
		sfmin = int(arr_read[3])
		vfmax = int(arr_read[4])
		vfmin = int(arr_read[5])
		efsize = int(arr_read[6])
		dfsize = int(arr_read[7])
		hgmax = int(arr_read[8])
		hgmin = int(arr_read[9])
		sgmax = int(arr_read[10])
		sgmin = int(arr_read[11])
		vgmax = int(arr_read[12])
		vgmin = int(arr_read[13])
		egsize = int(arr_read[14])
		dgsize = int(arr_read[15])
		hbmax = int(arr_read[16])
		hbmin = int(arr_read[17])
		sbmax = int(arr_read[18])
		sbmin = int(arr_read[19])
		vbmax = int(arr_read[20])
		vbmin = int(arr_read[21])
		ebsize = int(arr_read[22])
		dbsize = int(arr_read[23])
		hnmax = int(arr_read[24])
		hnmin = int(arr_read[25])
		snmax = int(arr_read[26])
		snmin = int(arr_read[27])
		vnmax = int(arr_read[28])
		vnmin = int(arr_read[29])
		ensize = int(arr_read[30])
		dnsize = int(arr_read[31])
		fblur = int(arr_read[32])
		f.close
		#`print '%d'%(sfmin)
	print 'loaded'
	return None

def nothing(x):
	pass

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(goal_top_left, goal_bottom_left) = leftMost
	D = dist.cdist(goal_top_left[np.newaxis], rightMost, "euclidean")[0]
	(goal_bottom_right, goal_top_right) = rightMost[np.argsort(D)[::-1], :]
	return np.array([goal_top_left, goal_top_right, goal_bottom_right, goal_bottom_left], dtype="float32")

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
	#im = cv2.imread("images/acak%d.jpg"%iteration)
	#im = cv2.imread("/home/barelangfc/Foto_BolaPutih/my_photo-%d.jpg"%iteration)
	#im = cv2.imread("/home/barelangfc/python/my_photo-1.jpg")
	#print("images/acak%d.jpg"%iteration)

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
	dilate_parsefield = cv2.dilate(parsefield, kernel, iterations = dfsize)
	#erode_parsefield = cv2.erode(parsefield,(20,20),iterations = efsize)
	#dilate_parsefield = cv2.dilate(erode_parsefield,(20,20),iterations = dfsize)

	#detection goal
	goal_colorConv_image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
	g_lower_val = np.array([hgmin,sgmin,vgmin])
	g_upper_val = np.array([hgmax,sgmax,vgmax])
	parsegoal = cv2.inRange(goal_colorConv_image,g_lower_val,g_upper_val)
	erode_parsegoal = cv2.erode(parsegoal,kernel,iterations = egsize)
	dilate_parsegoal = cv2.dilate(erode_parsegoal,kernel,iterations = dgsize)

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
	erode_parseball_invert_field = cv2.erode(parseball_invert_field,kernel,iterations = ensize) #5
	dilate_parseball_invert_field = cv2.dilate(erode_parseball_invert_field,kernel,iterations = dnsize)
	#print '%d'%(ensize)

	ball_colorConv_image = cv2.cvtColor(field_image,cv2.COLOR_BGR2YUV)
	b_lower_val = np.array([hbmin,sbmin,vbmin])
	b_upper_val = np.array([hbmax,sbmax,vbmax])
	parseball_white = cv2.inRange(ball_colorConv_image,b_lower_val,b_upper_val)
	erode_parseball_white = cv2.erode(parseball_white,kernel,iterations = ebsize) #2
	dilate_parseball_white = cv2.dilate(erode_parseball_white,kernel,iterations = dbsize) #2
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
			ball_iteration = 1;
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
					if selected_ball == ball_iteration:
						print 'Ball Number %d ==> X = %d Y = %d W = %d H = %d Radius = %d Area = %.2f R_Area = %.2f Area_Rat = %.2f WH_Rat = %.2f Percent_Wh = %.2f'%(ball_iteration,ball_centre_x,ball_centre_y,ball_width,ball_height,ball_radius,ball_area,ball_rect_area,ball_area_ratio,ball_wh_ratio, percent_white)
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
				ball_iteration += 1;

		# Ball detection mode 2
		# mode 2 digunakan khusus bola dekat yg tdk bisa terdeteksi oleh mode 1
		# tambah deteksi lokasi center terhadap warna hitam lapangan
		#if field_w !=0 and field_h != 0:
	if (ball_found == False and mod2 == False and mod1 == False) or (mod2 == True and mod1 == False) :
		if len(b_wh_contours) > 0:
			b_wh_sorted_contours = sorted(b_wh_contours, key=cv2.contourArea, reverse=True)[:5]
			ball_found =  False
			ball_iteration = 1;
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
					if selected_ball == ball_iteration:
						print 'Ball Number %d ==> X = %d Y = %d W = %d H = %d Radius = %d Area = %.2f R_Area = %.2f Area_Rat = %.2f WH_Rat = %.2f Percent_Wh = %.2f'%(ball_iteration,ball_centre_x,ball_centre_y,ball_width,ball_height,ball_radius,ball_area,ball_rect_area,ball_area_ratio,ball_wh_ratio, percent_white)
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
				ball_iteration += 1;

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
		goal_iteration = 1;
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
				if selected_goal == goal_iteration :
					print 'Goal Num %d ==> X = %d Y = %d W = %d H = %d Area = %.2f R_Area = %.2f Area_Rat = %.2f WH_Rat = %.2f'%(goal_iteration, goal_centre_x,goal_centre_y,goal_width,goal_height,goal_area,goal_rect_area,goal_area_ratio,goal_wh_ratio)
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
			goal_iteration += 1;
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
		iteration += 1
	elif k == ord('a'):
		iteration -= 1

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
