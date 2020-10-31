# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#               File Name : getPointsPositionVerticalCase.py                                    = #
#                                                                                               = #
#                          Programmer : Luo Meng                                                = #
#                                                                                               = #
#                         Start Date : 08/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 08, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    getPointsPosition -- get the positions of points suit the condition                        = #
# =============================================================================================== #

import numpy as np


# get the positions of points suit the condition
class getPointsPosition:
    def __init__(self):
        # Initialize lists to store value of images
        self.image3 = []
        self.image4_shift = []
        self.image5 = []
        self.image6_shift = []
    
        # intialize an array to save case of increasing slope 
        self.array_index_forward_new = [[] * i for i in range(256)]
        # intialize an array to save case of decreasing slope
        self.array_index_backward_new = [[] * i for i in range(256)]
        
        # intialize lists to store index of perfect match, with one error and more than one error case.
        self.list1 = []
        self.list2 = []
        self.list3 = []
        self.list4 = []
        self.list5 = []
        self.list6 = []
        
        self.num1 = 0
        self.num2 = 0
        self.num3 = 0
        self.num4 = 0
        self.num5 = 0
        self.num6 = 0
        
        # # Initialize and store the total number of the case of increasing slope 
        self.sum1 = 0
        # Initialize and store the total number of the case of decreasing slope
        self.sum2 = 0
    
    
    def mainProgram(self):
        self.readData()
        # get suitable points of case of increasing slope and decrease case 
        self.getSuitableSlopePoints()
        
        # we need to compare the corresponding pixel in two figures here.
        # case of increasing slope
        self.num1, self.num2, self.num3, self.list1, self.list2, self.list3 = self.imageDataAnalyse(self.image3, self.image4_shift, self.array_index_forward_new)
        
        # case of decreasing slope
        self.num4, self.num5, self.num6, self.list4, self.list5, self.list6 = self.imageDataAnalyse(self.image3, self.image4_shift, self.array_index_backward_new)
    
        # display data, normally we could anotate this
        self.displayDataOnScreen() 
        
        self.saveData()
    
    
    # read data from .npy file
    def readData(self):
        self.image3 = np.load("C:/Users/15025/Desktop/IPHT/data/image3.npy").T
        self.image4_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image4_shift.npy").T
        self.image5 = np.load("C:/Users/15025/Desktop/IPHT/data/image5.npy").T
        self.image6_shift = np.load("C:/Users/15025/Desktop/IPHT/data/image6_shift.npy").T


    def getSuitableSlopePoints(self):        
        # we choose three pixels, so the index of last pixel is:
        pixel_num = 256 - 3
        
        # intialize an array to save case of increasing slope 
        array_index_forward = [[] * i for i in range(256)]
        # intialize an array to save case of decreasing slope
        array_index_backward = [[] * i for i in range(256)]
        
        # for-loop for line cuts
        for i in range(256):
            # for-loop for different pixel points
            for j in range(pixel_num):
                if self.image5[i][j] < self.image5[i][j + 1] < self.image5[i][j + 2]:
                    array_index_forward[i].append(j)
                if self.image5[i][j] > self.image5[i][j + 1] > self.image5[i][j + 2]:
                    array_index_backward[i].append(j)
        
        # Then we need to judage a big slope, if the slope is big, we choose it
        # In order to get a proper number of points, we should try different threshold value.
        # set the threshold of the case of increasing slope, if bigger than this value, we choose it
        _threshold_value1 = 0.019
        
        # set the threshold of the case of decreasing slope, if bigger than this value, we choose it
        _threshold_value2 = 0.019
          
        for i,list1 in enumerate(array_index_forward):
            for j in list1:
                if ((self.image5[i][j + 2] - self.image5[i][j + 1]) / 2 + (self.image5[i][j + 1] - self.image5[i][j]) / 2) / 2 > _threshold_value1:
                    # add suitable value to our list
                    self.array_index_forward_new[i].append(j)
                    # count total number
                    self.sum1 += 1
        
        for i,list1 in enumerate(array_index_backward):
            for j in list1:                
                if ((self.image5[i][j] - self.image5[i][j + 1]) / 2 + (self.image5[i][j + 1] - self.image5[i][j + 2]) / 2)  / 2 > _threshold_value2:
                    # add suitable value to our list
                    self.array_index_backward_new[i].append(j)
                    # count total number
                    self.sum2 += 1
        
        
    # we calcualte 3 pixels in one time, try to find some regulation
    def imageDataAnalyse(self, image3, image4, index_array1):
        # num1: the positon of image suits the index(+-)1
        # num2: the positon of image suits the index(+-)2
        # num3: other situation, 3 or more
        _num1 = 0
        _num2 = 0
        _num3 = 0
        index_list1 = []
        index_list2 = []
        index_list3 = []
        
        for value,indexlist in enumerate(index_array1):
            for value1, index in enumerate(indexlist):
                _image3_11 = image3[value][index + 1]
                _image3_21 = image3[value][index + 2]
                _image3_31 = image3[value][index + 3]
                _image3 = image3[value][index]
                _image3_1 = image3[value][index - 1]
                _image3_2 = image3[value][index - 2]
                _image3_3 = image3[value][index - 3]
                _image4_11 = image4[value][index + 1]
                _image4_21 = image4[value][index + 2]
                _image4_31 = image4[value][index + 3]
                _image4 = image4[value][index]
                _image4_1 = image4[value][index - 1]
                _image4_2 = image4[value][index - 2]
                _image4_3 = image4[value][index - 3]
                
                if index == 255:
                    if (_image3 > _image3_1 or (_image3 < image3_1 and _image3_1 > image3_2)) and (_image4 > _image4_1 or (_image4_1 > _image4_2 and image4_1 > _image4)):
                        _num1 += 1
                        index_list1.append([value, index])
                    elif (_image3_2 > _image3_1 and _image3_2 > _image3_3) and (image4_2 > image4_1 and image4_2 > image4_3):
                        _num2 += 1
                        index_list2.append([value, index])
                    else:
                        _num3 += 1
                        index_list3.append([value, index])
                        
                elif index == 254:
                    if ((_image3 > _image3_1 and _image3 > _image3_11) or (_image3_11 > _image3) or (_image3 < _image3_1 and _image3_1 > _image3_2)) and ((_image4 > _image4_1 and _image4 > _image4_11) or (_image4_11 > _image4) or (_image4_1 > _image4_2 and _image4_1 > _image4)):
                        _num1 += 1
                        index_list1.append([value, index])
                    elif (_image3_2 > _image3_1 and _image3_2 > _image3_3) and (_image4_2 > _image4_1 and _image4_2 > _image4_3):
                        _num2 += 1
                        index_list2.append([value, index])
                    else:
                        _num3 += 1
                        index_list3.append([value, index])
                
                 
                # generate case
                else:            
                    if ((_image3 > _image3_1 and _image3 > _image3_11) or (_image3_11 > _image3 and _image3_11 > _image3_21) or (_image3 < _image3_1 and _image3_1 > _image3_2)) and ((_image4 > _image4_1 and _image4 > _image4_11) or (_image4_11 > _image4 and _image4_11 > _image4_21) or (_image4_1 > _image4_2 and _image4_1 > _image4)):
                        _num1 += 1
                        index_list1.append([value, index])
                    elif ((_image3_2 > _image3_1 and _image3_2 > _image3_3) or (_image3_21 > _image3_11 and _image3_21 > _image3_31)) and ((_image4_2 > _image4_1 and _image4_2 > _image4_3) or (_image4_21 > _image4_11 and _image4_21 > _image4_31)):
                        _num2 += 1
                        index_list2.append([value, index])
                    else:
                        _num3 += 1
                        index_list3.append([value, index])
                
        return _num1, _num2, _num3, index_list1, index_list2, index_list3
        
    
    # print data
    def displayDataOnScreen(self):
        print()
        print(f"the total number matching the condition is: {self.sum1}")
        print(f"the prefectly matching numer is: {self.num1}")
        print(f"number of error 1 is: {self.num2}")
        print(f"number of more error is: {self.num3}")
        percentage_num1 = (self.num1 / self.sum1) * 100
        percentage_num2 = (self.num2 / self.sum1) * 100
        percentage_num3 = (self.num3 / self.sum1) * 100
        print(f"the percentage of perfectly matching is {percentage_num1}")
        print(f"the percentage of error 1 is {percentage_num2}")
        print(f"the percentage of error more than one is {percentage_num3}")
        
        print("\n" * 2)
        
        print(f"the total number matching the condition is: {self.sum2}")
        print(f"the prefectly matching numer is: {self.num4}")
        print(f"number of error 1 is: {self.num5}")
        print(f"number of more error is: {self.num6}")
        percentage_num1 = (self.num4 / self.sum2) * 100
        percentage_num2 = (self.num5 / self.sum2) * 100
        percentage_num3 = (self.num6 / self.sum2) * 100
        print(f"the percentage of perfectly matching is {percentage_num1}")
        print(f"the percentage of error 1 is {percentage_num2}")
        print(f"the percentage of error more than one is {percentage_num3}")
    
    
    # save data to a .txt file, then next we just need to read it from this .txt file
    def saveData(self):
        with open("C:/Users/15025/Desktop/IPHT/data/image3PixelVertical.txt", "w") as f:
            f.write(str(self.list1))
            f.write("\n")
            f.write(str(self.list2))
            f.write("\n")
            f.write(str(self.list3))
            f.write("\n")
            f.write(str(self.list4))
            f.write("\n")
            f.write(str(self.list5))
            f.write("\n")
            f.write(str(self.list6))
            
            
if __name__ == "__main__":
    main = getPointsPosition()
    main.mainProgram()