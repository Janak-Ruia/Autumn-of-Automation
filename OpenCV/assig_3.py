import cv2
import numpy as np 
import matplotlib.pyplot as plt 
'''
img = cv2.imread("janak.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
edge = cv2.Canny(img, 10, 100)
f = plt.figure(figsize=(10,10))
ax1=f.add_subplot(1,2,1)
ax1.imshow(img)
ax1.set_title('original')
ax2=f.add_subplot(1,2,2)
ax2.imshow(edge)
ax2.set_title('Pencil sketch')
plt.show()
'''
def find_edges(img, minval, maxval):
	edge=cv2.Canny(img, minval, maxval)
	return edge

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	edges = find_edges(frame, 80, 120)
	cv2.imshow("Pencil sketch", edges)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()