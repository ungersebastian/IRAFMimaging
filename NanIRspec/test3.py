# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 12:53:14 2022

@author: DELL
"""
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import os

path_project = path.dirname(path.realpath(__file__))
path_final = path.join(path_project, r'resources\BacillusSubtilis\2108_BacVan30_1400-1659cm-1')
headerfile = 'BacVan30_0011.txt'

my_data = ir(path_final, headerfile)
pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = np.reshape(hyPIRFwd['data'], (1,hyPIRFwd['data'].shape[0],hyPIRFwd['data'].shape[1], hyPIRFwd['data'].shape[2]))

pos =  [my_file['Caption']=='TopographyFwd' for my_file in my_data['files']]
topo = np.array(my_data['files'])[pos][0]
topo = topo['data']

intim = np.sum(data, axis = -1)[0]

plt.imsave('temp.png',intim)
img = mpimg.imread('temp.png')
os.remove('temp.png')

X,Y = np.meshgrid(*(np.arange(s) for s in topo.shape))

#Plotting
fig = plt.figure(figsize = [5, 5])
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, topo, rstride=1, cstride=1, facecolors=img, linewidth=0, antialiased=False, shade=False)

#%%

"""
data = pd.read_csv('BacVan30_0011hyPIRFwdWavelengths.txt', skiprows=[0], delimiter="\t")
wave = data.iloc[:,0]
Inten = data.iloc[:,1]

#Third dimension (Should be color / height)
def Fn (d, g):
    return np.sqrt((d**2)+(g**2))


#Specify the limits as the minimum & maximum of the spectrum
xmin = np.min(wave)
ymin = np.min(Inten)
xmax = np.max(wave)
ymax = np.max(Inten)


x = np.linspace(xmin, xmax, 32)
y = np.linspace(ymin, ymax, 32)
X, Y = np.meshgrid(x, y)
z = Fn(X, Y)

#Coloring
k = LightSource(270, 45)
rgb = k.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')

#Plotting
fig = plt.figure(figsize = [30, 16])
ax = fig.add_subplot(111, projection='3d')
#The dimensions should be between wave and Intens with the third dimenstion, but it gives mis-matching to the shape !
ax.plot_surface(X, Y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)



"""