# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#                File Name : draw3DFigureWithNormalization.py                                   = #
#                                                                                               = #
#                          Programmer : Luo Meng                                                = #
#                                                                                               = #
#                         Start Date : 07/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 07, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    Draw3DFigure -- 3D figure plot without normalization                                       = #
# =============================================================================================== #

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


class Draw3DFigure:
    def __init__(self):
        self.image3 = []
    
    
    def mainProgram(self):
        self.readData()
        self.draw3DFigure()
        
        
    def readData(self):
        self.image3 = np.load("C:/Users/15025/Desktop/IPHT/data//image3.npy")

    
    def draw3DFigure(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = np.arange(-128, 128)
        Y = np.arange(-128, 128)
        x, y = np.meshgrid(X, Y)
        
        # Plot the surface.
        surf = ax.plot_surface(x, y, self.image3, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        
        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.show()


if __name__ == "__main__":
    main = Draw3DFigure()
    main.mainProgram()
