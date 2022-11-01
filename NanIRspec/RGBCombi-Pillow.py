# -*- coding: utf-8 -*-
"""
Created on Wed Jun 2022

Combine 3 images from 3 channels R, G, B and overly them
"""

import numpy as np
from PIL import Image

Img1 = Image.open('N2200_Au0002_PiF.tiff')
Img2 = Image.open('N2200_Au0003_PiF.tiff')
Img3 = Image.open('N2200_Au0004_PiF.tiff')


def RGBCombi (Img1, Img2, Img3):

    Array1 = np.array(Img1)
    ImgG = Array1.astype(np.uint8())

    Array2 = np.array(Img2)
    ImgR = Array2.astype(np.uint8())

    Array3 = np.array(Img3)
    ImgB = Array3.astype(np.uint8())

    SumCombi = np.add(ImgG, ImgR)
    SumCombi2 = np.add(SumCombi, ImgB)


    ComRGB = Image.fromarray(SumCombi2)
    r, g, b = ComRGB.split()

    merge = Image.merge('RGB', (b, r, g))
    merge.show()
    #merge.save("RGBPILpif.tiff")


RGBCombi (Img1, Img2, Img3)

