import cv2
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('janak.jpg', -1)
img_og = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_og, cv2.COLOR_RGB2GRAY)
thresh, img_bnw = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig=plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)
ax1=fig.add_subplot(1,3,1)
ax2=fig.add_subplot(1,3,2)
ax3=fig.add_subplot(1,3,3)
ax1.imshow(img_og)
ax2.imshow(img_gray, cmap="gray")
ax3.imshow(img_bnw, cmap="gray")
ax1.title.set_text("original")
ax2.title.set_text("Gray")
ax3.title.set_text("Black and White")
plt.show()

cap = cv2.VideoCapture(0)

def red_to_blue(img):
	img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	red_low=np.array([1, 80, 100])
	red_high=np.array([25, 255, 255])
	mask=cv2.inRange(img_hsv, red_low, red_high)
	img[mask>0]=(255,0,0)
	return img

while(True):

    ret, frame = cap.read()
    img=red_to_blue(frame)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
