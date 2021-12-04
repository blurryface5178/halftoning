# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:19:05 2021

@author: Manisha Das
"""
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

### Dehalftoning

#Importing Image

#from PIL import Image
#im = Image.open('First_PanAm_Flight_Havana.tif')
#im.show()
#I = plt.imread('First_PanAm_Flight_Havana.tif')

image1 = cv2.imread('First_PanAm_Flight_Havana.tif',0)
#plt.imshow(image1) 
# image2 = cv2.imread('Kahanamoku_Jan-37_cropped_bitmap.tif',0)
# image3 = cv2.imread('Leuteritz portrait_1200dpi.tif',0)
# image4 = cv2.imread('PA Air Ways March 31_3200 dpi.tif',0)
# f = plt.figure(figsize=(12,5))
# f.add_subplot(221),plt.imshow(image1, cmap = 'gray')
# f.add_subplot(222),plt.imshow(image2, cmap = 'gray')
# f.add_subplot(223),plt.imshow(image3, cmap = 'gray')
# f.add_subplot(224),plt.imshow(image3, cmap = 'gray')
# #plt.savefig("inputs.png", bbox_inches="tight")
# plt.show()



#task 1
### its too slow so/crap, resize it to 64x64

def dft(input_img,c1,c2):
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    output_img = np.zeros((rows,cols),complex)
    for m in range(0,rows):
        for n in range(0,cols):
            for x in range(0,rows//2):
                for y in range(0,cols//2):
                    output_img[m][n] += c1*c2*input_img[x][y] * np.exp(-1j*2*math.pi*(m*x/rows+n*y/cols))
    return output_img

c1=1
c2=1


#dft_man = dft(image1 ,c1,c2)
#plt.subplot(121),plt.imshow(image1,'gray'),plt.title('Original Halftoned Image')
#plt.subplot(122),plt.imshow(dft_man,'gray'),plt.title('dft_manual_output')
#plt.show()
#plt.savefig("inputs.png", bbox_inches="tight")

#Task 2
#out_dft = np.log(np.abs(dft_man))
#plt.subplot(121),plt.imshow(out_dft,'gray'),plt.title('dft_manual_output')
#plt.show()
 

fft_np = np.fft.fft2(image1)
fshift = np.fft.fftshift(fft_np)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.figure(figsize=(12, 10))
plt.subplot(121),plt.imshow(image1, cmap = 'gray')
plt.title('Halftoned Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()



