# -*- coding: utf-8 -*-
"""
hyPirana is a program for analysis of PiFM hyPIR spectra acquired using VistaScan
Created on Fri Apr 24 08:06:26 2020

@author: ungersebastian

modified on Fri Oct 16 by Daniela Taeuber for application to the spectral range of one tuner only
modified by Mohammad Soltaninezhad for rescaling intensities and saving figures
last modified on Fri June 11 by Daniela Taeuber: re-arrangement of structural elements & commenting

hyPirana.py can do:
- read a hyperspectral data set from a Vistascope
- use AreaSelect for selecting a region of interest in the data via a GUI
- use a substrate spectrum for calibrating the hyperspectral data. The substrate text-file should have one line with column titles and consist of two entries (wavelengths and intensities) separated by tabs "\t"
- calculate mean spectra
- run a PCA on the data set

For application to two seperate spectral ranges, see "hyPIRana_splitRange.py"
"""

#%% imports & parameters
#import pandas as pd
from os.path import join
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from AreaSelect import getArea

from datetime import datetime
import importlib as il
if False:  # to conserve order (which gets swirled up by pep)
    sys.path.append(
        "r//mars/usr/FA8_Mikroskopie/FAG82_BiomedizinischeBildgebung/BioPOLIM/PiFM/")
    import MicroPy as mipy

from IRAFM import IRAFM as ir
"""#####################################################"""
## Daniela: links to resources
#path_import = r'PiFM/Retina/200229_Ret24'
#headerfile = 'Ret240033.txt'
#path_import = r'PiFM/Retina/200405_Ret29'
#headerfile = 'Ret29r20006.txt'
#path_dir = r'//mars/usr/FA8_Mikroskopie/FAG82_BiomedizinischeBildgebung/BioPOLIM/'
#pathsub = '/Ret24_CaF_2001_Tuner1349-1643.txt'
"""#####################################################"""
## Mohammad: links to resources
#pathsub = "F:/daniela/retina/NanIRspec/resources/CaF20001mean1349-1643.txt"
#path_import = r'F:\daniela\retina\NanIRspec\resources'
#headerfile = 'Ret240012.txt'
#path_dir = r'retina/NanIRspec/resources'
"""#####################################################"""
## Robin: links to resources
#path_import = r'210514BacVan30/BacVan_30'
#headerfile = 'BacVan30_0013.txt'
#path_dir = r'C:/Users/Robin/Documents/HS Aalen/Masterarbeit/Daten/'
#pathsub = 'BacVan30_CaF2_av_Tuner2.txt'
"""#####################################################"""

#path_final = join(path_dir, path_import)
#path_substrate = join(path_final, pathsub)
today = datetime.strftime(datetime.now(), "%Y%m%d")!
#save_path = join(path_final, today, '/')
#save_path = path_final + '/' + '200405_Ret29Results' + '/'
#save_path = path_final + '/'

"""#####################################################"""
## Dev-Sebastian: hotlink to resources for testing purposes
path_project = path.dirname(path.realpath(__file__))
path_final = path.join(path_project, r'resources')
headerfile = 'Ret240012.txt'
"""#####################################################"""

if 0:
    il.reload(mipy)

if 0:
    il.reload(mipy)

#%%
my_data = ir(path_final, headerfile) 

pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = np.reshape(hyPIRFwd['data'], (hyPIRFwd['data'].shape[0]*hyPIRFwd['data'].shape[1], hyPIRFwd['data'].shape[2]))

my_sum = np.sum(data, axis = 1)
data_train = data[my_sum != 0]

#my_data.plot_all()

from utils_chemometrics import pca, norm
from utils_plot import stack_plot, pairs

my_pca = pca(
    data = data_train, 
    nComp = 3, # integer==number of pc components, float (0,1)== number of pc components which yields that amount of variance explained
    center = True,
    pNorm = 1, # p-Norm to this power (even odd numbers), also possible: infinity-normalization using np.infty
    normAxis = 1
    )

scores, loadings = my_pca.scores, my_pca.components

stack_plot(loadings)
d = pairs(scores)
#%% functions

def mean_spec(my_sum, coord):
    '''
    Calculates and plots mean spectra with standard deviations
    '''
    mean_spc = np.mean(spc_norm, axis = 0)
    std_spc = np.std(spc_norm, axis = 0)

    my_fig = plt.figure()
    ax = plt.subplot(111)
    ax.fill_between(x = my_data['wavelength'], y1 = mean_spc+std_spc, y2 = mean_spc-std_spc, alpha = 0.6)
    plt.gca().invert_xaxis() #inverts values of x-axis
    ax.plot(my_data['wavelength'], mean_spc)
    ax.set_xlabel('wavenumber ['+my_data['PhysUnitWavelengths']+']')
    ax.set_ylabel('intensity (normalized)')

    ax.set_yticklabels([])
    plt.title('mean spectrum')

    my_fig.tight_layout()

#%% loads data and plots associated VistaScan parameter images

my_data = ir(path_final, headerfile) 

my_data.plot_all()


# establish save-path
#dir_test_existance(save_path)

#%% Part1

testIm = my_data['files'][0]
data = testIm['data']

my_p, my_a = getArea(image = data, delta = 3)

pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = hyPIRFwd['data']
data_select = data[my_a == 1]   # here are now just the selected spectra
# print("dataselectshape",data_select.shape)
# print("datashpe",data.shape)


#%% checks validity of data and sorts them
w,h=my_a.shape
my_spc = my_data.return_spc()*np.reshape(my_a,newshape=(w*h,1))
# np.savetxt("myspcrescale.txt",my_spc,delimiter="\t")
print("my_spc",my_spc.shape)
print(my_spc.shape)
print(my_data.return_spc().shape)
print(my_a.shape)
my_wl  = my_data['wavelength']
# np.savetxt("WaVeLeNgTh.txt",my_wl,delimiter="\t")
pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = np.reshape(hyPIRFwd['data'], (hyPIRFwd['data'].shape[0]*hyPIRFwd['data'].shape[1], hyPIRFwd['data'].shape[2]))
my_sum = np.sum(my_spc, axis = 1)
coord = np.arange(len(my_sum))
zeros = np.zeros(len(my_sum))
data = data[my_sum != 0]
coord = coord[my_sum != 0]
my_sum = my_sum[my_sum != 0]


spc_norm = np.array([spc/s for spc, s in zip(data, my_sum)])
print("spcnorm",spc_norm.shape)

#%%Rescaling
#pathsub='/Ret24_CaF_2001_Tuner1349-1643.txt'
#path_substrate = join(path_final, pathsub)

b=len(spc_norm)

print("b",b)
# np.savetxt("myspcraw.txt",my_spc,delimiter="\t")

# print("myw1",my_wl.shape)
with open( path_substrate, 'r' ) as fopen:
    y_sub = np.array(fopen.readlines())
y_sub = [''.join(l.split('\n')) for l in y_sub][1:]
y_sub = (np.array([l.split('\t') for l in y_sub]).T).astype(float)
y_sub=y_sub[1,:]
print("ysub",y_sub.shape)
for i in range( b ):
    spc_norm[i, :] = spc_norm[i, :] / y_sub
#rescaling finished


#%%
# PlotIt
mean_spc = np.mean(spc_norm, axis = 0)
std_spc = np.std(spc_norm, axis = 0)
print("mean spc",mean_spc.shape)
print("std spc",std_spc.shape)
my_fig = plt.figure()
ax = plt.subplot(111)
ax.fill_between(x = my_data['wavelength'], y1 = mean_spc+std_spc, y2 = mean_spc-std_spc, alpha = 0.6)
plt.gca().invert_xaxis() #inverts values of x-axis
ax.plot(my_data['wavelength'], mean_spc)
ax.set_xlabel('wavenumber ['+my_data['PhysUnitWavelengths']+']')
ax.set_ylabel('intensity (normalized)')
ax.set_yticklabels([])
plt.title('mean spectrum')
my_fig.savefig( 'mean spectrum.png' )
my_fig.tight_layout()

#save data as text
mypath = join(save_path,'1meanspec.txt')
np.savetxt(mypath, mean_spc)
mypath = join(save_path, '1myspectrum.txt')
np.savetxt(mypath, my_a)
mypath = join(save_path, '1mypoint.txt')
np.savetxt(mypath, my_p)
mypath = join(save_path, '1stdspec.txt')
np.savetxt(mypath,std_spc)

#%%


#%%
from sklearn.decomposition import PCA

ncomp = 3


model = PCA(n_components=ncomp)

transformed_data = model.fit(spc_norm-mean_spc).transform(spc_norm-mean_spc).T
loadings = model.components_
stack_plot(loadings)


#%%

#%%
print("loadingtype",type(model))
print("mysum",my_sum.shape)
print("data",data.shape)
print("spc_norm",spc_norm.shape)
print("meanspc",mean_spc.shape)
print("std",std_spc.shape)
print("transform",transformed_data.shape)
print("loading",loadings.shape)
my_fig = plt.figure()
ax = plt.subplot(111)
plt.gca().invert_xaxis() #inverts values of x-axis
for icomp in range(ncomp):
    ax.plot(my_data['wavelength'], loadings[icomp], label='PC'+str(icomp+1) )
ax.set_xlabel('wavenumber ['+my_data['PhysUnitWavelengths']+']')
ax.set_ylabel('intensity (normalized)')
ax.set_yticklabels([])
ax.legend()
plt.title('PCA-Loadings')
my_fig.savefig( 'PCA-Loadings.png' )
my_fig = plt.figure()
ax = plt.subplot(111)
ax.plot(transformed_data[0], transformed_data[1], '.')
ax.set_xlim(np.quantile(transformed_data[0], 0.05),np.quantile(transformed_data[0], 0.95))
ax.set_ylim(np.quantile(transformed_data[1], 0.05),np.quantile(transformed_data[1], 0.95))
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.title('scatterplot')
my_fig.savefig( 'scatterplot.png' )
my_fig.tight_layout()

maps = [zeros.copy() for icomp in range(ncomp)]
for icomp in range(ncomp):
    maps[icomp][coord] = transformed_data[icomp]
    maps[icomp] = np.reshape(maps[icomp], (my_data['xPixel'], my_data['yPixel']) )
    
    my_fig= plt.figure()
    ax = plt.subplot(111)
    plt.colorbar( ax.imshow(maps[icomp], cmap = 'coolwarm', extent = my_data.extent()) )
    
    ax.set_xlabel('x scan ['+my_data['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data['YPhysUnit']+']')
    plt.title('factors PC'+str(icomp+1))
    my_fig.savefig( 'factors PC.png' )

    my_fig.tight_layout()

plt.show()




