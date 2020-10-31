# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#             File Name : draw3DFigureWithoutNormalization.py                                   = #
#                                                                                               = #
#                          Programmer : Luo Meng                                                = #
#                                                                                               = #
#                         Start Date : 07/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 07, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    ViewFigures -- display figures by using nip(NanoImagingPack)                               = #
# =============================================================================================== #

import numpy as np
import NanoImagingPack as nip


class ViewFigures:
    def __init__(self):
        self.image3 = []
        self.image4_shift = []
        self.image5 = []
        self.image6_shift = []
    
    
    # main program
    def mainProgram(self):
        self.readData()
        self.displayFigureByNIP()
        
    
    # read data from .npy files
    def readData(self):
        self.image3 = np.load("C:/Users/15025/Desktop/IPHT/data/image3.npy")
        self.image4_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image4_shift.npy")
        self.image5 = np.load("C:/Users/15025/Desktop/IPHT/data/image5.npy")
        self.image6_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image6_shift.npy")

    
    def displayFigureByNIP(self):
        nip.v5(nip.catE(self.image3, self.image4_shift, self.image5, self.image6_shift))
        # could do different combination here


if __name__ == "__main__":
    main = Draw3DFigure()
    main.mainProgram()
