# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:53:10 2022

@author: DELL
"""

import cv2
import numpy as np


image1 = cv2.imread('N2200_Au0002_PiF.tiff')
image2 = cv2.imread('N2200_Au0003_PiF.tiff')
image3 = cv2.imread('N2200_Au0004_PiF.tiff')

b1, g1, r1 = cv2.split(image1)
b2, g2, r2 = cv2.split(image2)
b3, g3, r3 = cv2.split(image3)

b = np.add(b1, b2)
Blue = np.add(b, b3)

g = np.add(g1, g2)
Green = np.add(g, g3)

r = np.add(r1, r2)
Red = np.add(r, r3)

#cv2.show("Blue", b)
#cv2.show("Green", g)
#cv2.show("Red", r)

Merge = cv2.merge([Green, Blue, Red])
#cv2.imshow("RGB", Merge)
cv2.imwrite("RGBpif.tiff", Merge)
