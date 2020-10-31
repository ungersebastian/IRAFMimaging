# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#                       File Name : plotLineCuts.py                                             = #
#                                                                                               = #
#                          Programmer : Luo Meng                                                = #
#                                                                                               = #
#                         Start Date : 06/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 06, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    drawAllLineCuts -- draw all line cuts of two PiFM images and one topo image                = #
# =============================================================================================== #

import numpy as np
import matplotlib.pyplot as plt


class drawAllLineCuts:
    def __init__(self):
        self.image3 = []
        self.image4_shift = []
        self.image5 = []
        self.image6_shift = []
    
    
    def mainProgram(self):
        self.readData()
        self.drawAndSaveLineCuts()
        
        
    def readData(self):
        self.image3 = np.load("C:/Users/15025/Desktop/IPHT/data//image3.npy")
        self.image4_shift = np.load("C:/Users/15025/Desktop/IPHT/data//image4_shift.npy")
        self.image5 = np.load("C:/Users/15025/Desktop/IPHT/data//image5.npy")
        self.image6_shift = np.load("C:/Users/15025/Desktop/IPHT/data//image6_shift.npy")


    def drawAndSaveLineCuts(self):
        # set plot style
        plt.style.use('fivethirtyeight')
        
        # 256 is the size of one line cut, is also the number of line cuts
        x = np.arange(256)
        for i in x:  
            plt.figure()
            
            # the other two PiFM images
            # plt.plot(x, image1[i], label='image1')
            # plt.plot(x, image2[i], label='image2')
            
            # plot two PiFM line cuts and one topo line cut in one figure
            plt.plot(x, self.image3[i], label='PiFM image at 880 cm^-1')
            plt.plot(x, self.image4_shift[i], label='PiFM image at 1360 cm^-1')
            plt.plot(x, self.image5[i], label='Topo image at 880 cm^-1')
            
            # add legend
            plt.legend()
            
            # use tight layout
            plt.tight_layout()
            
            # display figure, normally we annotate this
            # plt.show()
            
            # save figure
            plt.savefig("C:/Users/15025/Desktop/IPHT/slliceFigure(2 together)/{}.png".format(i))


if __name__ == "__main__":
    main = drawAllLineCuts()
    main.mainProgram()