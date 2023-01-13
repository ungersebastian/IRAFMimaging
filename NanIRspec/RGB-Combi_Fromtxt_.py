# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:40:11 2022

Combine RGB image from 3 pixel data of 3 images

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


#Reading the data
Data1 = pd.read_csv('Topo2.txt', delimiter=('\t'), header = None)
Data2 = pd.read_csv('Topo3.txt', delimiter=('\t'), header = None)
Data3 = pd.read_csv('Topo4.txt', delimiter=('\t'), header = None)

def RGBCmbi(Data1, Data2, Data3):
    #Reshaping arrays of the images
    Arr = np.array(np.ones(shape = (1,1,3)))

    Green = np.array(Data1)
    Gmax = np.max(Green)
    Greens = (Green / Gmax)
    Greens = Greens[:, :, np.newaxis]
    G = np.multiply(Arr, Greens)

    Red = np.array(Data2)
    Rmax = np.max(Red)
    Reds = (Red / Rmax)
    Reds = Reds[:, :, np.newaxis]
    R = np.multiply(Arr, Reds)

    Blue = np.array(Data3)
    Bmax = np.max(Blue)
    Blues = (Blue / Bmax)
    Blues = Blues[:, :, np.newaxis]
    B = np.multiply(Arr, Blues)

    #plotting R, G, B images
    plt.subplot(141)
    R[:, :, [1,2]] = 0
    RedOnly = plt.imshow(R)

    plt.subplot(142)
    G[:, :, [0,2]] = 0
    GreenOnly = plt.imshow(G)

    plt.subplot(143)
    B[:, :, [0,1]] = 0
    BlueOnly = plt.imshow(B)


    #Combination and Overlay
    Comb = np.add(R, G)
    Combi = np.add(Comb, B)

    plt.subplot(144)
    CombiRGB = plt.imshow(Combi)

    rgb = plt.figure()
    rgb = plt.imshow(Combi)
    
    #Saving Combi as txt file:
    Combi_reshaped = Combi.reshape(Combi.shape[0], -1)
    np.savetxt('Combi.txt', Combi_reshaped, delimiter = ('\t'))
    
    
    return R, G, B
    
    
RGBCmbi(Data1, Data2, Data3)
