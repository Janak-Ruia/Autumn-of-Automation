import cv2
import numpy as np 

cap = cv2.VideoCapture("messi.mp4")
cap2 = cv2.VideoCapture(0)

def find(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (9,9), 0)
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,1000, param1=200, param2=10, minRadius=1, maxRadius=60)
	return circles

while True:
	ret, frame=cap.read()
	if frame is not None:
		c = find(frame)
	else: 
		print("Frame is none")
		break
	if c is not None:
		circles = np.uint16(c)
		for i in circles[0,:]:
			frame_fin = cv2.circle(frame, (i[0], i[1]), i[2], (0,0,255), 5)
		cv2.imshow("Detected Circles", frame_fin)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
'''

def format(img):
	img = cv2.GaussianBlur(img, (3,3), 0)
	lower = (0, 0, 100)
	higher =(0,0,255)
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	mask = cv2.inRange(hsv, lower, higher)
	fin = cv2.bitwise_and(img , img, mask=mask)
	#img[mask>0]=(255,255,255)
	#img[mask==0]=(0,0,0)
	
	return fin

while True:
	ret, frame = cap.read()
	mask = format(frame)
	#cnt =cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cv2.imshow("dfsdf", mask)
	cv2.imshow("real", frame)
	
	if cnt is None:
		print("No contours")
		break
	c = max(cnt, key=cv2.contourArea)
	((x, y), radius) = cv2.minEnclosingCircle(c)
	M = cv2.moments(c)
	center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	if radius>100:
		cv2.circle(frame, (int(x), int(y)), int(radius), (0,0,255), 4)
	cv2.imshow("fin", frame)
	
	if cv2.waitKey(2) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
'''