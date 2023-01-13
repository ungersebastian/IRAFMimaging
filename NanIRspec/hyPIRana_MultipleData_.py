# -*- coding: utf-8 -*-
"""
hyPirana is a program for analysis of PiFM hyPIR spectra acquired using VistaScan
Created on Fri Apr 24 08:06:26 2020

@author: ungersebastian

Last modified on Fri Oct 16 by Daniela Taeuber for application to the spectral range of one tuner only
Modified by Maryam Ali to work on multiple datasets 
"""
#%% imports & parameters

from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import importlib as il
if False:  # to conserve order (which gets swirled up by pep)
    sys.path.append(
        "r//mars/usr/FA8_Mikroskopie/FAG82_BiomedizinischeBildgebung/BioPOLIM/PiFM/")
    import MicroPy as mipy

from pifm_image import pifm_image
from IRAFM import IRAFM as ir

#Loading the data
path_import33 = r'C:\.......'
headerfile33 = 'Ret240033.txt'
path_import20 = r'C:\......'
headerfile20 = 'Ret240020.txt'
path_import12 = r'C:......'
headerfile12 = 'Ret240012.txt'


path_dir = r'C:\Users\DELL\Documents\Python Works\Data Analysis'
path_final33 = join(path_dir, path_import33)
path_final20 = join(path_dir, path_import20)
path_final12 = join(path_dir, path_import12)
today = datetime.strftime(datetime.now(), "%Y%m%d")
#save_path = path_final + today + '/' #does not work !
#save_path = join(path_final, today, '/')
#save_path = path_final + '/' + '200405_Ret29Results' + '/'
save_path = path_final20 + '/'

if 0:
    il.reload(mipy)

if 0:
    il.reload(mipy)


#%% loads data and plots associated VistaScan parameter images

my_data12 = pifm_image(path_final12, headerfile12) 
my_data20 = pifm_image(path_final20, headerfile20) 
my_data33 = pifm_image(path_final33, headerfile33) 

#If wavelength range isn't matching and didn't work, use the IRAFM
#my_data12 = ir(path_final12, headerfile12) 
#my_data20 = ir(path_final20, headerfile20) 
#my_data33 = ir(path_final33, headerfile33) 


my_data12.plot_all()
my_data20.plot_all()
my_data33.plot_all()



#%% checks validity of data and sorts them

#Calibration
Calib_file = pd.read_csv('Ret24_CaF_2001_Tuner1349-1643.txt', delimiter = '\t')
Cali_data = np.array(Calib_file)
Cali_Spc = Cali_data[:,1]

my_spc12 = my_data12.return_spc()
my_spc20 = my_data20.return_spc()
my_spc33 = my_data33.return_spc()

my_wl  = my_data12['wavelength']




pos12 =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data12['files']]
hyPIRFwd12 = np.array(my_data12['files'])[pos12][0]
data12 = np.reshape(hyPIRFwd12['data'], (hyPIRFwd12['data'].shape[0]*hyPIRFwd12['data'].shape[1], hyPIRFwd12['data'].shape[2]))
my_sum12 = np.sum(data12, axis = 1)


pos20 =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data20['files']]
hyPIRFwd20 = np.array(my_data20['files'])[pos20][0]
data20 = np.reshape(hyPIRFwd20['data'], (hyPIRFwd20['data'].shape[0]*hyPIRFwd20['data'].shape[1], hyPIRFwd20['data'].shape[2]))
my_sum20 = np.sum(data20, axis = 1)

pos33 =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data33['files']]
hyPIRFwd33 = np.array(my_data33['files'])[pos33][0]
data33 = np.reshape(hyPIRFwd33['data'], (hyPIRFwd33['data'].shape[0]*hyPIRFwd33['data'].shape[1], hyPIRFwd33['data'].shape[2]))
my_sum33 = np.sum(data33, axis = 1)


AllData = np.array([[data12], [data20], [data33]])
AllDatas = np.squeeze(AllData, 1).T
#AllDatas20 = AllDatas[:, :, 1]
my_sum = np.sum(AllDatas, axis = 0)

#my_sums = np.squeeze(my_sum, 1).T
#my_sums = my_sums[my_sums != 0]


coord12 = np.arange(len(my_sum12))
zeros12 = np.zeros(len(my_sum12))
data12 = data12[my_sum12 != 0]
coord12 = coord12[my_sum12 != 0]
my_sum12 = my_sum12[my_sum12 != 0]


coord20 = np.arange(len(my_sum20))
zeros20 = np.zeros(len(my_sum20))
data20 = data20[my_sum20 != 0]
coord20 = coord20[my_sum20 != 0]
my_sum20 = my_sum20[my_sum20 != 0]


coord33 = np.arange(len(my_sum33))
zeros33 = np.zeros(len(my_sum33))
data33 = data33[my_sum33 != 0]
coord33 = coord33[my_sum33 != 0]
my_sum33 = my_sum33[my_sum33 != 0]


coord = np.arange(len(my_sum))
zeros = np.zeros(len(my_sum))
#AllData = AllData[my_sum != 0]
#coord = coord[my_sum != 0]
my_sum = my_sum[my_sum != 0]

spc_norm12 = np.array([(spc)/s for spc, s in zip(data12, my_sum12)])
spc_norm20 = np.array([(spc)/s for spc, s in zip(data20, my_sum20)])
spc_norm33 = np.array([(spc)/s for spc, s in zip(data33, my_sum33)])



spc_norm= np.array([(spc/Cali_Spc)/s for spc, s in zip(AllDatas.T, my_sum)])
spc_norm2 = spc_norm.T.reshape(spc_norm.T.shape[0], -1)
#spc_norm2 = spc_norm2.T
my_sum = np.sum(spc_norm2, axis = 0)
coord = np.arange(len(my_sum))
zeros = np.zeros(len(my_sum))
#AllData = AllData[my_sum != 0]
#coord = coord[my_sum != 0]
my_sum = my_sum[my_sum != 0]



#%%
# PlotIt
mean_spc12 = np.mean(spc_norm12, axis = 0)
std_spc12 = np.std(spc_norm12, axis = 0)

mean_spc20 = np.mean(spc_norm20, axis = 0)
std_spc20 = np.std(spc_norm20, axis = 0)

mean_spc33 = np.mean(spc_norm33, axis = 0)
std_spc33 = np.std(spc_norm33, axis = 0)

mean_spc = np.mean(spc_norm, axis = 1).T

my_fig = plt.figure()
ax = plt.subplot(111)

plt.gca().invert_xaxis() #inverts values of x-axis

ax.plot(my_data12['wavelength'], mean_spc)


ax.set_xlabel('wavenumber ['+my_data12['PhysUnitWavelengths']+']')
ax.set_ylabel('intensity (normalized)')
ax.set_yticklabels([])
plt.title('mean spectrum')
my_fig.tight_layout()

#save data as text
#mypath = join(save_path,'meanspec.txt')
#np.savetxt(mypath, mean_spc)

mean_spc2 = mean_spc[:, 0]
mean_spc2 = mean_spc2.T

#%%

from sklearn.decomposition import PCA
ncomp = 2
model = PCA(n_components=ncomp)



transformed_data = model.fit(spc_norm2.T - mean_spc2).transform(spc_norm2.T - mean_spc2).T
Twelve = transformed_data[:, 0:1024]
Twenty = transformed_data[:, 1024:2048]
ThirtyThree = transformed_data[:, 2048:3072]

loadings = model.components_



my_fig = plt.figure()
ax = plt.subplot(111)
plt.gca().invert_xaxis() #inverts values of x-axis
for icomp in range(ncomp):
    ax.plot(my_data12['wavelength'], loadings[icomp], label='PC'+str(icomp+1) )
    ax.legend()
ax.set_xlabel('wavenumber ['+my_data20['PhysUnitWavelengths']+']')
ax.set_ylabel('intensity (normalized)')
ax.set_yticklabels([])
plt.title('PCA-Loadings')



my_fig = plt.figure()
ax = plt.subplot(111)
ax.plot(transformed_data[0], transformed_data[1], '.')
ax.set_xlim(np.quantile(transformed_data[0], 0.05),np.quantile(transformed_data[0], 0.95))
ax.set_ylim(np.quantile(transformed_data[1], 0.05),np.quantile(transformed_data[1], 0.95))
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.title('scatterplot')
my_fig.tight_layout()

maps = [zeros.copy() for icomp in range(ncomp)]
maps12 = [zeros12.copy() for icomp in range(ncomp)]
maps20 = [zeros20.copy() for icomp in range(ncomp)]
maps33 = [zeros33.copy() for icomp in range(ncomp)]

for icomp in range(ncomp):
    
    maps[icomp][coord] = transformed_data[icomp]
    maps[icomp] = np.reshape(maps[icomp],(32,96))
    
    my_fig = plt.figure()
    ax = plt.subplot(111)
    plt.colorbar( ax.imshow(maps[icomp], cmap = 'coolwarm' , vmin = -2000, vmax = 10000))
    ax.set_xlabel('x scan ['+my_data20['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data20['YPhysUnit']+']')
    ax.legend()
    plt.title('factors PC'+str(icomp+1))
    my_fig.tight_layout()
    
    
    maps12 = maps[icomp][:, 0:32]
      
    
    my_fig = plt.figure()
    ax = plt.subplot(111)
    plt.colorbar( ax.imshow(maps12, cmap = 'coolwarm' , vmin = -2000, vmax = 10000))
    ax.set_xlabel('x scan ['+my_data20['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data20['YPhysUnit']+']')
    ax.legend()
    plt.title('240012:factors PC'+str(icomp+1))
    my_fig.tight_layout()
    g =  maps[icomp][:, 0:32].reshape(maps[icomp][:, 0:32].shape[0], -1)
    np.savetxt('Twelve-PC'+str(icomp+1)+'.txt', g, delimiter = '\t')

    
    maps20 = maps[icomp][:, 32:64]
    
    my_fig = plt.figure()
    ax = plt.subplot(111)
    plt.colorbar( ax.imshow(maps20, cmap = 'coolwarm' , vmin = -2000, vmax = 10000))
    ax.set_xlabel('x scan ['+my_data20['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data20['YPhysUnit']+']')
    ax.legend()
    plt.title('240020:factors PC'+str(icomp+1))
    my_fig.tight_layout()
    g2 =  maps[icomp][:, 32:64].reshape(maps[icomp][:, 32:64].shape[0], -1)
    np.savetxt('Twenty-PC'+str(icomp+1)+'.txt', g2, delimiter = '\t')

    
    maps33 = maps[icomp][:, 64:96 ]
        
    
    my_fig = plt.figure()
    ax = plt.subplot(111)
    plt.colorbar( ax.imshow(maps33, cmap = 'coolwarm' , vmin = -2000, vmax = 10000))
    ax.set_xlabel('x scan ['+my_data20['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data20['YPhysUnit']+']')
    ax.legend()
    plt.title('240033:factors PC'+str(icomp+1))
    my_fig.tight_layout()
    g3 =  maps[icomp][:, 64:96].reshape(maps[icomp][:, 64:96].shape[0], -1)
    np.savetxt('Thirty-PC'+str(icomp+1)+'.txt', g3, delimiter = '\t')
 


