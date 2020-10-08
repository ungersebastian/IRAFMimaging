# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#              File Name : drawSlopePointsOnToPoVerticalCase.py                                 = #
#                                                                                               = #
#                          Programmer : Luo Meng                                                = #
#                                                                                               = #
#                         Start Date : 08/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 08, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    SlopeFigure -- display points suits condition of slope on topo figure                      = #
# =============================================================================================== #

# import module we need
import ast
import numpy as np
import matplotlib.pyplot as plt 


class SlopeFigure:
    def __init__(self):
        # intialize list to store data
        self.list1 = []
        self.list2 = []
        self.list3 = []
        self.list4 = []
        self.list5 = []
        self.list6 = []
        
        self.image3 = []
        self.image4_shift = []
        self.image5 = []
        self.image6_shift = []
    
    
    def mainProgram(self):
        self.readData()
        self.drawFigures()
    
    
    # read data from txt file
    def readData(self):
        # read data from txt file 
        with open("C:/Users/15025/Desktop/IPHT/data/image3PixelVertical.txt", "r") as f:
            self.list1 = ast.literal_eval(f.readline())
            self.list2 = ast.literal_eval(f.readline())
            self.list3 = ast.literal_eval(f.readline())
            self.list4 = ast.literal_eval(f.readline())
            self.list5 = ast.literal_eval(f.readline())
            self.list6 = ast.literal_eval(f.readline())
            
        # read data from .npy files
        # here we only need 4 data
        self.image3 = np.load("C:/Users/15025/Desktop/IPHT/data/image3.npy")
        self.image4_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image4_shift.npy")
        self.image5 = np.load("C:/Users/15025/Desktop/IPHT/data/image5.npy")
        self.image6_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image6_shift.npy")


    # draw figure
    def drawFigures(self):
        # case of increasing slope
        plt.figure(1)
        plt.imshow(self.image5)
        for value in self.list1:
            scatter1 = plt.scatter(value[1], value[0], marker="o", color='red')
            
        for value in self.list2:
            scatter2 = plt.scatter(value[1], value[0], marker="^", color='blue')
        
        plt.title("slope of increase")
        plt.legend((scatter1, scatter2), ("perfect match", "1 pixel error"))
        plt.show()
        
        # case of decreasing slope
        plt.figure(2)
        plt.imshow(self.image5)
        for value in self.list4:
            scatter1 = plt.scatter(value[1], value[0], marker="o", color='red')
            
        for value in self.list5:
            scatter2 = plt.scatter(value[1], value[0], marker="^", color='blue')
        
        plt.title("slope of decrease") 
        plt.legend((scatter1, scatter2), ("perfect match", "1 pixel error")) 
        plt.show()


if __name__ == "__main__":
    main = SlopeFigure()
    main.mainProgram()