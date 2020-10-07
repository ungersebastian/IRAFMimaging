# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#                       File Name : saveImageByNIP.py                                           = #
#                                                                                               = #
#                          Programmer : Luo Meng                                                = #
#                                                                                               = #
#                         Start Date : 07/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 07, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    saveImage -- save image by using nip(NanoImagingPack)                                      = #
# =============================================================================================== #

import numpy as np
import NanoImagingPack as nip


class saveImage:
    def __init__(self):
        self.image3 = []
        self.image4_shift = []
        self.image5 = []
        self.image6_shift = []
    
    
    # main program
    def mainProgram(self):
        self.readData()
        self.saveImageByNIP()
        
    
    # read data from .npy files
    def readData(self):
        self.image3 = np.load("C:/Users/15025/Desktop/IPHT/data/image3.npy")
        self.image4_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image4_shift.npy")
        self.image5 = np.load("C:/Users/15025/Desktop/IPHT/data/image5.npy")
        self.image6_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image6_shift.npy")
        
    
    # save data by using NIP
    def saveImageByNIP(self):
        # save image 3
        img = nip.image(self.image3)
        mypath = r'C:/Users/15025/Desktop/IPHT/posterImage/PiFM at 880 cm-1.tif'
        nip.imsave(img, mypath, rescale=True, BitDepth=16, Floating=False, truncate=True, rgb_tif = False)
        
        # # save image4_shift
        img = nip.image(self.image4_shift)
        mypath = r'C:/Users/15025/Desktop/IPHT/posterImage/PiFM at 1360 cm-1.tif'
        nip.imsave(img, mypath , rescale=True, BitDepth=16, Floating=False, truncate=True, rgb_tif = False)
        
        # # save image5
        img = nip.image(self.image5)
        mypath = r'C:/Users/15025/Desktop/IPHT/posterImage/Topography at 880 cm-1.tif'
        nip.imsave(img, mypath , rescale=True, BitDepth=16, Floating=False, truncate=True, rgb_tif = False)
        
        # # save image6_shift
        img = nip.image(self.image6_shift)
        mypath = r'C:/Users/15025/Desktop/IPHT/posterImage/Topography at 1360 cm-1.tif'
        nip.imsave(img, mypath, rescale=True, BitDepth=16, Floating=False, truncate=True, rgb_tif = False)


if __name__ == "__main__":
    main = saveImage()
    main.mainProgram()