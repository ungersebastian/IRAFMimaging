# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:39:32 2022

@author: DELL
"""

from PIL import Image
import numpy as np


img1 = Image.open('pic11.png').convert("RGBA")
pix1 = img1.load()
img2 = Image.open('pic12.png').convert("RGBA")
pix2 = img2.load()

mask = float()

x1, y1 = img1.size
x2, y2 = img2.size
        
#for i, j in range (x1, y1):
 #   for k, z in range (x2, y2):
  #    if pix1[i, j] == pix2[k,z]:
   #       mask =  0.5 
    #  else:
     #    mask =  0
         
         
new_image = Image.blend(img1, img1, alpha = 0.5)


new_image.show()

