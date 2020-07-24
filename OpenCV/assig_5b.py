import cv2
import numpy as np 

img = cv2.imread("ball.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture("messi.mp4")
cap2 = cv2.VideoCapture(0)

sift = cv2.ORB_create(1000)

#detection
kp1, det1 = sift.detectAndCompute(img,None)
det1 = np.asarray(det1, np.float32)
#feature matching
index_param = dict(algorithm = 0, trees = 5)
search_param = dict()
flann = cv2.FlannBasedMatcher(index_param, search_param)

while True:
    _, frame = cap.read()
    gframe=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gframe is not None:
        kp_g, det_g = sift.detectAndCompute(gframe, None)
        det2 =  np.asarray( det_g, np.float32) 
        matches = flann.knnMatch(det1, det2, k=2)
        good=[]
        for m, n in matches:
            if m.distance<0.8*n.distance:
                good.append(m)

        #img3 = cv2.drawMatches(img, kp1, gframe, kp_g, good, gframe)

        if len(good)>7:
            
            query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_g[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            # Perspective transform
            h, w = img.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, matrix)
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Can't Find", gframe)

            
        #cv2.imshow("feed", gframe)
        #cv2.imshow("reference", img)
        #cv2.imshow("img3", img3)
    key=cv2.waitKey(1)
    if key == 27:
        break


cv2.waitKey(0)
cv2.destroyAllWindows()