# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#                     File Name : drawTwoPiFMImage.py                                           = #
#                                                                                               = #
#                          Programmer : Luo Meng                                                = #
#                                                                                               = #
#                         Start Date : 07/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 07, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    drawPiFMImages -- draw two topo images with a common colorbar                              = #
# =============================================================================================== #

import numpy as np
import NanoImagingPack as nip
import matplotlib.pyplot as plt


class drawPiFMImages:
    def __init__(self):
        # initialize a list to store data of first topo image
        self.image3 = []
        # initialize a list to store data of second topo image
        self.image4_shift = []
    
    
    # main program
    def mainProgram(self):
        self.readData()
        self.displayPiFMImage()
        
    
    # read data from .npy files
    def readData(self):
        self.image3 = np.load("C:/Users/15025/Desktop/IPHT/data/image3.npy")
        self.image4_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image4_shift.npy")


    def displayPiFMImage(self):
        # display PiFM image at 880 cm^-1
        plt.figure(1)
        plt.imshow(self.image3, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(ticks=[])
        
        # display PiFM image at 1360 cm^-1
        plt.figure(2)
        plt.imshow(self.image4_shift, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(ticks=[])


if __name__ == "__main__":
    main = drawPiFMImages()
    main.mainProgram()