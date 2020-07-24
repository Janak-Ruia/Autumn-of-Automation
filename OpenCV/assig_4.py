import cv2
import numpy as np 

img_og = cv2.imread("test2.png")

gray = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (7,7))
ret, thresh = cv2.threshold(gray,120,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU )
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#fin = cv2.drawContours(img_og, contours, -1, (0,255,0), 1)


#ignore the words and boundary
cont =[]
m=[]
for c in contours:
	l=cv2.arcLength(c, True)
	if cv2.arcLength(c, True)>200 and cv2.arcLength(c, True)<600:
		cont.append(c)
		m.append(cv2.moments(c))
center=[]
for moment in m:
	center.append((int(moment['m10']/moment['m00']),int(moment['m01']/moment['m00'] )))


print(len(cont))
print(center)

fin2 = cv2.drawContours(img_og, cont, -1, (0,0,255), 1)
for c in center:
	cv2.circle(fin2, c, 5, (255,0,0), -1)

#M=cv2.moments(contours[1])
#cx = int(M['m10']/M['m00'])
#cy = int(M['m01']/M['m00'])
#print(M)
#print((cx, cy))
#fin[cx,cy]=(255,255,255)
#cv2.imshow("all", fin)
cv2.imshow("selected", fin2)
cv2.waitKey(0)
cv2.destroyAllWindows()