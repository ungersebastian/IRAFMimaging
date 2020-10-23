# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#                     File Name : dataFormatTransfer.py                                         = #
#                                                                                               = #
#                        Programmer : René Lachmann                                             = #
#                                                                                               = #
#                         Start Date : 06/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 06, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    DataFormatTrasnform -- transfer data from .int format to .npy format                       = #
# =============================================================================================== #

# import modules we need
import NanoImagingPack as nip
import numpy as np


class DataFormatTrasnform:
    def __init__(self):     
        # set common path of data we want to read
        self.common_path = r"C:/Users/15025/Desktop/IPHT/Program/program(ProvidedByDaniela)/irafmimaging-master/IRAFim/Tests/191127_NP_correlation/"
        
        # set individual path of data we want to read
        self.image_list_PiFM = ['Per-PDAGA-NP0007PiFFwd.int','Per-PDAGA-NP0008PiFFwd.int','Per-PDAGA-NP0009PiFFwd.int','Per-PDAGA-NP0010PiFFwd.int','Per-PDAGA-NP0009Topo OutFwd.int', 'Per-PDAGA-NP0010Topo OutFwd.int'] 
        
        # initialize data_type, later use it to set type of data we read
        self.data_type = np.dtype(np.int32) 
        
        # initialize a blank list, later use it to store data
        self.image_stack = []
        
        # initialize lists to store each data
        # PiFM images
        self.image1 = []
        self.image2 = []
        self.image3 = []
        self.image4 = []
        # ToPo images: we set image5(topo 9) as reference one
        self.image5 = []
        self.image6 = []
        
        # initialize lists to store shifted image
        # Topo images
        self.image1_shift = []
        self.image2_shift = []
        self.image4_shift = []
        # PiFM image
        self.image6_shift = []
        
        # distance of shift, we need to calculate this first.
        # the shift between topo 7 and topo 9
        self.shift_79 = np.array([-0.86, -1.96])
        # the shift between topo 8 and topo 9
        self.shift_89 = np.array([0.072, -0.756])
        # the shift between topo 10 and topo 9
        self.shift_109 = np.array([-0.34, -0.07])
    
    def mainProgram(self):
        self.readData()
        self.normaliseData()
        self.shiftImage()
        self.saveData()
        
    
    # read PiFM data from original .int file
    def readData(self):
        for num in range(len(self.image_list_PiFM)):
            with open(self.common_path + self.image_list_PiFM[num], 'rb') as f:
                self.image_stack.append(f.read())
                news = [int(np.sqrt(len(self.image_stack[num]) / 4)),] * 2
                self.image_stack[num] = np.reshape(np.frombuffer(self.image_stack[num], self.data_type), news)
    
        # Topo images
        self.image1 = self.image_stack[0]
        self.image2 = self.image_stack[1]
        self.image3 = self.image_stack[2]
        self.image4 = self.image_stack[3]
        # PiFM images
        self.image5 = self.image_stack[4]
        self.image6 = self.image_stack[5]


    # normalise data
    def normaliseData(self):
        # Topo images
        self.image1 = self.image1 / np.max(self.image1)
        self.image2 = self.image2 / np.max(self.image2)
        self.image3 = self.image3 / np.max(self.image3)
        self.image4 = self.image4 / np.max(self.image4)
        # PiFM images
        self.image5 = self.image5 / np.max(self.image5)
        self.image6 = self.image6 / np.max(self.image6)

    
    # shift image
    def shiftImage(self):   
        # Topo images
        self.image1_shift = nip.shift(self.image1, -self.shift_79, dampOutside=True)
        self.image2_shift = nip.shift(self.image2, -self.shift_89, dampOutside=True)
        self.image4_shift = nip.shift(self.image4, -self.shift_109, dampOutside=True)
        # PiFM image
        self.image6_shift = nip.shift(self.image6, -self.shift_109, dampOutside=True)
            
    
    # save data in .npy file
    def saveData(self):
        # the directory could be changed
        np.save("C:/Users/15025/Desktop/IPHT/data/image3.npy", self.image3)
        np.save("C:/Users/15025/Desktop/IPHT/data/image4_shift.npy",self.image4_shift)
        np.save("C:/Users/15025/Desktop/IPHT/data/image5.npy", self.image5)
        np.save("C:/Users/15025/Desktop/IPHT/data/image6_shift.npy", self.image6_shift)


if __name__ == "__main__":
    main = DataFormatTrasnform()
    main.mainProgram()
