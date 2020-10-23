# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#                     File Name : drawTwoToPoImage.py                                           = #
#                                                                                               = #
#                          Programmer : Luo Meng                                                = #
#                                                                                               = #
#                         Start Date : 07/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 07, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    drawTopoImages -- draw two topo images with a common colorbar                              = #
# =============================================================================================== #

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


class drawTopoImages:
    def __init__(self):
        # initialize a list to store data of first topo image
        self.image5 = []
        # initialize a list to store data of second topo image
        self.image6_shift = []
    
    
    # main program
    def mainProgram(self):
        self.readData()
        self.convertImage()
        self.displayTopoImage()
        
    
    # read data from .npy files
    def readData(self):
        self.image5 = np.load("C:/Users/15025/Desktop/IPHT/data/image5.npy")
        self.image6_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image6_shift.npy")
        
    
    def convertImage(self):
        self.image5 = 1 - self.image5
        self.image6_shift = 1 - self.image6_shift


    def displayTopoImage(self):
        # set canvas
        fig = plt.figure()
        
        # create subplot with 1 row 2 columns
        (ax1, ax2) = ImageGrid(fig, 111,          
                          nrows_ncols=(1,2),
                          share_all=True,
                          cbar_location="right",
                          cbar_mode="single",
                          # cbar_size="7%",
                          cbar_pad=0.15,
                          )
        
        # plot first topo image in position ax1
        im = ax1.imshow(self.image5, cmap='gray')
        # plot image5 in position ax1
        im = ax2.imshow(self.image6_shift, cmap='gray')
        
        # do not display x ticks in subplot figure 1
        ax1.set_xticks([])
        # do not display y ticks in subplot figure 1
        ax1.set_yticks([])
        # do not display x ticks in subplot figure 2
        ax2.set_xticks([])
        # do not display y ticks in subplot figure 2
        ax2.set_yticks([])
        
        # draw color bar for these two subplot fugures
        # actually for subplot figure 1, but they have same value range
        ax1.cax.colorbar(im, ticks=[])


if __name__ == "__main__":
    main = drawTopoImages()
    main.mainProgram()
