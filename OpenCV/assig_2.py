import cv2
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('test.png')

f_scale=plt.figure()
ax=f_scale.add_subplot(1,1,1)
#ax.imshow(img)
#-----------scaling----------------------#
resized = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
img = resized[370:530, 240:420]
ax.imshow(img)
ax.set_title('Scaled by factor of 2 along x and y, then cropped')
plt.show()


#----------------translation------------------------------#
rows = img.shape[0]
cols=img.shape[1]
M1 = np.float32([[1,0,50],[0,1,50]])
M2 = np.float32([[1,0,-50],[0,1,50]])
M3 = np.float32([[1,0,50],[0,1,-50]])
M4 = np.float32([[1,0,-50],[0,1,-50]])
shifted1 = cv2.warpAffine(img,M1,(cols,rows))
shifted2 = cv2.warpAffine(img,M2,(cols,rows))
shifted3 = cv2.warpAffine(img,M3,(cols,rows))
shifted4 = cv2.warpAffine(img,M4,(cols,rows))
f_translate = plt.figure(figsize=(15,15))
f_translate.suptitle("Translations", fontsize=16)
ax1 = f_translate.add_subplot(2,2,1)
ax1.imshow(shifted1)
ax2 = f_translate.add_subplot(2,2,2)
ax2.imshow(shifted2)
ax3 = f_translate.add_subplot(2,2,3)
ax3.imshow(shifted3)
ax4 = f_translate.add_subplot(2,2,4)
ax4.imshow(shifted4)
plt.show()
#-----------------rotation-------------------------------#
M=[]
rotated=[]
for i in range(1, 5):
	M.append(cv2.getRotationMatrix2D((cols//2, rows//2), 90*i, 1))
	rotated.append(cv2.warpAffine(img, M[i-1], (cols, rows)))

f_rot = plt.figure(figsize=(15,15))
f_rot.suptitle("Rotations", fontsize=16)
ax=[]
for i in range(1,5):
	ax.append(f_rot.add_subplot(2,2,i))
	ax[i-1].imshow(rotated[i-1])
plt.show()
#---------------Afine-Transformation-------------------------#
pts1 = np.float32([[120,260],[120,190],[210,190]])
pts2 = np.float32([[150,260],[100,190],[210,190]])

M = cv2.getAffineTransform(pts1,pts2)

a_t = cv2.warpAffine(img,M,(cols,rows))



#--------------------Perspective-------------------#

pts1=np.float32([[120, 260], [120, 190], [210, 190], [210, 260]])
pts2=np.float32([[0,300], [0, 0], [300, 0], [300, 300]])
N = cv2.getPerspectiveTransform(pts1,pts2)


f_at = plt.figure(figsize=(15,15))
ax1 = f_at.add_subplot(1,2,1)
ax1.imshow(img)
ax1.set_title('Original')

ax2 = f_at.add_subplot(1,2,2)
ax2.imshow(a_t)
ax2.set_title('Affine Transform')


#-----------------------Blur--------------------#
f_blur = plt.figure(figsize=(15,15))
ax1=f_blur.add_subplot(2,2,1)
ax1.imshow(img)
ax1.set_title('Original')
blur=cv2.blur(img, (7,7))
#cv2.imshow("blur", blur)
ax2=f_blur.add_subplot(2,2,2)
ax2.imshow(blur)
ax2.set_title('Averaging - kernel = (7x7)')

gauss=cv2.GaussianBlur(img, (7,7), 0)
ax3=f_blur.add_subplot(2,2,3)
ax3.imshow(gauss)
ax3.set_title('Gaussian Filtering - kernel = (7x7)')

bilat=cv2.bilateralFilter(img, 9, 75, 75, cv2.BORDER_DEFAULT)
ax4=f_blur.add_subplot(2,2,4)
ax4.imshow(bilat)
ax4.set_title('Bilateral Fitering')

f_blur.suptitle('Smoothening', fontsize=16)
plt.show()
