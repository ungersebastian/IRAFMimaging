# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#                    File Name : useGuiDisplayLineCuts.py                                       = #
#                                                                                               = #
#                          Programmer : Luo Meng                                                = #
#                                                                                               = #
#                         Start Date : 08/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 08, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    UseGuiDisplayLineCuts -- show line cuts with parameters by using tinter interface          = #
# =============================================================================================== #

import os
import time
import numpy as np
from PIL import Image
from PIL import ImageTk
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
        

class UseGuiDisplayLineCuts:
    def __init__(self):
        # create a tk object
        self.tk = tk.Tk()
        
        # set title of gui interface
        self.tk.title("Figure dynamic show v1.02")
        
        # set size of gui interface
        self.tk.geometry("1100x700+350+150")
       
        self.data = None
        
        self.para1 = None
        self.para2 = None
        self.para3 = None
        
        self.label1 = None
        
        
    def mainProgram(self):
        self.data = Data()
        self.data.mainProgram()
        
        # run gui
        self.startGui()
        
        self.tk.mainloop()
        
        
    def startGui(self):
        basic_position_x = 700
        basic_position_y = 50
        
        self.para1 = tk.Entry(self.tk)
        self.para1.place(x=basic_position_x + 100, y=basic_position_y)
        self.para2 = tk.Entry(self.tk)
        self.para2.place(x=basic_position_x + 100, y=basic_position_y + 40)
        self.para3 = tk.Entry(self.tk)
        self.para3.place(x=basic_position_x + 100, y=basic_position_y + 80)
        
        tk.Label(self.tk, text="Set Parameters:").place(x=basic_position_x, y=basic_position_y - 30)
        tk.Label(self.tk, text="slice:").place(x=basic_position_x + 50, y=basic_position_y)
        tk.Label(self.tk, text="minX:").place(x=basic_position_x + 50, y=basic_position_y + 40)
        tk.Label(self.tk, text="maxX:").place(x=basic_position_x + 50, y=basic_position_y + 80)
        
        # at the begining show 50th line cut, it the figure has already exsited in the directory
        if os.path.exists(r"C:/Users/15025/Desktop/IPHT/figureSliceInInterval/1.png"):
            image = Image.open("C:/Users/15025/Desktop/IPHT/figureSliceInInterval/1.png")
            photo = ImageTk.PhotoImage(image)
            self.label1 = tk.Label(self.tk, image=photo)
            self.label1.image = photo
            self.label1.grid()
        ttk.Button(self.tk, text="plot", command=self.figurePlot).place(x=100,y=600)
        
        
    def figurePlot(self):
        # set non-GUI backend
        matplotlib.use('Agg')
        # get slice value, should int type
        slice1 = int(self.para1.get())
        # get value of left point of displaying interval 
        min_x = int(self.para2.get())
        # get value of right point of displaying interval
        max_x = int(self.para3.get())
        # set displaying interval by using value of min_x and max_x
        interval = np.arange(min_x, max_x)
        # set displaying style
        plt.style.use('fivethirtyeight')
        # create canvas
        plt.figure()
        # plot interval we want
        plt.plot(interval, self.data.image5[slice1][interval], label='ToPo_9')
        plt.plot(interval, self.data.image4_shift[slice1][interval], label='image4_shift')
        # set title of image
        plt.title("the {}th slice".format(slice1))
        # sex x label
        plt.xlabel("Coordinate")
        # set y label
        plt.ylabel("Intensity")
        # add legend
        plt.legend()
        # use tight layout style
        plt.tight_layout()
        # save figure to the director we want, later we take figure from same position to show on Gui interface
        plt.savefig("C:/Users/15025/Desktop/IPHT/figureSliceInInterval/{}.png".format(slice1))
        
        # we annotate this because we do not want this to show on another gui interface
        # plt.show()
        
        # sleep the program for waiting saving image, normally the speed of saving process is very fast, we do not need it
        # time.sleep(0.5)
        
        # before add new figure, we need to forget former one.
        if self.label1:
            self.label1.grid_forget()
        
        # read saved image and display on the Gui we create before
        image = Image.open(r"C:/Users/15025/Desktop/IPHT/figureSliceInInterval/{}.png".format(slice1))
        photo = ImageTk.PhotoImage(image)
        self.label1 = tk.Label(self.tk, image=photo)
        self.label1.image = photo
        self.label1.grid()


class Data:
    def __init__(self):
        self.image3 = []
        self.image4_shift = []
        self.image5 = []
        self.image6_shift = []
    
    
    # main program
    def mainProgram(self):
        self.readData()
        
    
    # read data from .npy files
    def readData(self):
        self.image3 = np.load("C:/Users/15025/Desktop/IPHT/data/image3.npy")
        self.image4_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image4_shift.npy")
        self.image5 = np.load("C:/Users/15025/Desktop/IPHT/data/image5.npy")
        self.image6_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image6_shift.npy")
        
        
if __name__ == '__main__':
    main = UseGuiDisplayLineCuts()
    main.mainProgram()