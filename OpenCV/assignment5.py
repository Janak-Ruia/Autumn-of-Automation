import cv2
import numpy as np 
detector = cv2.xfeatures2d.SURF_create()
min_matches=10
flannParam = dict(algorithm = 0, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})

img = cv2.imread('ball.jpg', 0)

#trying a mask
#mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)


train_kp, train_desc = detector.detectAndCompute(img, None) 

cam = cv2.VideoCapture(0)
vid = cv2.VideoCapture('messi.mp4')

lower_white = np.array([0,0,0], dtype=np.uint8)
upper_white = np.array([0,0,255], dtype=np.uint8)


while True:
	ret, Qimg_og =vid.read()
	Qimg = cv2.cvtColor(Qimg_og, cv2.COLOR_BGR2GRAY)
	#Qhsv = cv2.cvtColor(Qimg_og, cv2.COLOR_BGR2HSV)
	#Qimg = cv2.GaussianBlur(Qimg, (5,5), 0)
	#mask = cv2.inRange(Qhsv, lower_white, upper_white)
	Qkp, Qdesc = detector.detectAndCompute(Qimg, None)
	matches = flann.knnMatch(Qdesc, train_desc, k=2)
	good_matches=[]
	for m,n in matches:
		if m.distance<0.75*n.distance:
			good_matches.append(m)

	if len(good_matches)>min_matches:
		tp=[]
		qp=[]
		for m in good_matches:
			tp.append(train_kp[m.trainIdx].pt)
			qp.append(Qkp[m.queryIdx].pt)
		tp, qp = np.float32((tp, qp))
		H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
		h,w = img.shape
		trainBorder = np.float32([[[0,0], [0, h-1], [w-1,h-1], [w-1, 0]]])
		qBorder = cv2.perspectiveTransform(trainBorder, H)
		cv2.polylines(Qimg_og, [np.int32(qBorder)], True, (255,0,0), 5)
	else:
		print(f"Not enough matches {len(good_matches)}.")
	cv2.imshow('result', Qimg_og)
	if cv2.waitKey(10)==ord('q'):
		break
cam.release()
cv2.destroyAllWindows()
