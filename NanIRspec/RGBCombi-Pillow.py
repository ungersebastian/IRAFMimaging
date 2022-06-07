# -*- coding: utf-8 -*-
"""
Created on Wed Jun 2022
"""

import numpy as np
from PIL import Image



Img1 = Image.open('N2200_Au0002_PiF.tiff')
Array1 = np.array(Img1)
ImgG = Array1.astype(np.uint8())


Img2 = Image.open('N2200_Au0003_PiF.tiff')
Array2 = np.array(Img2)
ImgR = Array2.astype(np.uint8())


Img3 = Image.open('N2200_Au0004_PiF.tiff')
Array3 = np.array(Img3)
ImgB = Array3.astype(np.uint8())




SumCombi = np.add(ImgG, ImgR)
SumCombi2 = np.add(SumCombi, ImgB)


ComRGB = Image.fromarray(SumCombi2)
r, g, b = ComRGB.split()

merge = Image.merge('RGB', (b, r, g))
merge.show()
#merge.save("RGBPILpif.tiff")


