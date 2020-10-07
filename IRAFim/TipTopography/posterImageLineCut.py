# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#                   File Name : posterImageLineCut.py                                           = #
#                                                                                               = #
#                          Programmer : Luo Meng                                                = #
#                                                                                               = #
#                         Start Date : 07/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 07, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    posterImage -- draw line cut image of poster                                               = #
# =============================================================================================== #

import numpy as np
import matplotlib.pyplot as plt


class posterImage:
    def __init__(self):
        self.image3 = []
        self.image4_shift = []
        self.image5 = []
        self.image6_shift = []
    
    
    def mainProgram(self):
        self.readData()
        self.displayFigure()
        
        
    def readData(self):
        self.image3 = np.load("C:/Users/15025/Desktop/IPHT/data//image3.npy")
        self.image4_shift = np.load("C:/Users/15025/Desktop/IPHT/data//image4_shift.npy")
        self.image5 = np.load("C:/Users/15025/Desktop/IPHT/data//image5.npy")
        self.image6_shift = np.load("C:/Users/15025/Desktop/IPHT/data//image6_shift.npy")

    
    # draw line cuts(slice 23)
    def displayFigure(self):
        # set slice number, control which slice we want to show
        slice1 = 23
        
        # set x coordinate
        x = np.arange(256)
        
        # set plotting style
        plt.style.use('fivethirtyeight')
        # create canvas
        fig = plt.figure(1)
        # add subplot axis
        ax = fig.add_subplot()
        # draw slice 23
        plt.plot(x, self.image5[slice1], label='Topography at 880 cm^-1')
        plt.plot(x, self.image4_shift[slice1], label='PiFM at 1360 cm^-1')
        plt.plot(x, self.image3[slice1], label='PiFM at 880 cm^-1')
        
        # set position of rectangular
        upleft = (44, 0)
        # set width of rectangular
        width = 7
        # set height of rectangular
        height = 1
        # draw rectangular
        rect = plt.Rectangle(upleft, width, height, linewidth=2, edgecolor='black', facecolor='none')
        # add rect to figure
        ax.add_patch(rect)
        
        # draw arrow
        ax.annotate(s='large slope', xy=(51, 1), xytext=(60,1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05), size="xx-large"
                    )
        
        # set title of figure
        ax.set_title("line cut at y = 90 nm", size=25)
        
        # set a dict of font, later use it to set x label
        font_x_label = {"size": 25}
        
        # set properties of x label
        ax.set_xlabel("x (pixels) 1pixel ≈ 4 nm", font_x_label)
        
        # ax.set_ylabel("nromalized PiFM signal/normalised height", font_y_label)
        plt.text(x = -40, y = 0.2, s="nromalized PiFM signal", size=22, rotation=90)
        plt.text(x = -30, y = 0.27, s="normalized height", size=22, rotation=90, c="blue")
        
        # set font of ticks
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        
        # add legend to figure, and set its font
        plt.legend(loc=1, fontsize="x-large") 
        
        # display in a tight style
        plt.tight_layout()
        
        # display figure
        plt.show()
        
        # save figure, when not use, annotate this
        # plt.savefig("C:/Users/15025/Desktop/IPHT/figureSliceInInterval/{}.png".format(name))
        

if __name__ == "__main__":
    main = posterImage()
    main.mainProgram()
