import cv2
import numpy as np
import math
def transform(frame):
	#frame = resize(frame)
	try:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	except:
		pass
	frame = threshold(frame)
	frame = get_contours(frame)
	frame = resize(frame,100)
	return frame

def resize(frame,size):
	return cv2.resize(frame,(size,size))

def threshold(frame):
	_ , thresh = cv2.threshold(frame,120,255,cv2.THRESH_OTSU)

	return thresh

def get_contours(frame):
	se = np.ones((10,10),np.uint8)

	frame = cv2.erode(frame,se,iterations = 2)
	frame = cv2.dilate(frame,se,iterations = 2)
	frame = cv2.erode(frame,se,iterations = 1)
	
	(_,contours,_) = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	c = max(contours, key = cv2.contourArea)
	#largest_contour = [c]
	hull = [cv2.convexHull(c) for c in contours]
	x,y,w,h = cv2.boundingRect(c)
	#final = cv2.drawContours(frame,hull,-1,(255,255,255))
	final = frame.copy()
	#cv2.rectangle(final,(x,y),(x+w,y+h),(255,255,255),2)
	#cv2.imshow('ar',final)
	#cv2.waitKey(0)
	if w>h:
		dom = w
	else:
		dom = h
	bg = np.zeros((dom,dom),np.uint8)
	roi = final[y:y+h,x:x+w]
	image = bg
	if w>h:
		image[int((w-h)/2):int((w+h)/2),0:w]=roi
	else:
		image[0:h,int((h-w)/2):int((w+h)/2)]=roi
	
	return image

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

def write_text(frame,text):
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (50,50)
	fontScale              = 2
	fontColor              = (0,0,255)
	lineType               = 3

	cv2.putText(frame,text,bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
	return frame