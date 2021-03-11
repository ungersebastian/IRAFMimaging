# -*- coding: utf-8 -*-
"""
hyper is a program for analysis of PiFM hyPIR spectra acquired using VistaScan
Created on Fri Apr 24 08:06:26 2020

@author: ungersebastian

Modified on Thu Apr 28 by Daniela Taeuber
"""

#%% imports & parameters

from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from IRAFM import IRAFM

path_import = r'PiFM/Retina/200405_Ret29'
headerfile = 'Ret29r20006.txt'
path_dir = r'//mars/usr/FA8_Mikroskopie/FAG82_BiomedizinischeBildgebung/BioPOLIM/'

path_final = join(path_dir, path_import)


#%% functions

def mean_spectrum(spc_norm, my_data):
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
    return mean_spc

#%% loads data and plots associated VistaScan parameter images

my_data = IRAFM(path_final, headerfile) 

my_data.plot_all()


#%% Part1

from AreaSelect import AreaSelect
testIm = my_data['files'][0]
data = testIm['data']

c = AreaSelect(None, data)

#%% Part2
my_p=c.points
my_a=c.area


pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = hyPIRFwd['data']
data_select = data[my_a == 1]   # here are now just the selected spectra

#%% checks validity of data and sorts them

my_spc = my_data.return_spc()
my_wl  = my_data['wavelength']

split_wl = 1648

spc_low = my_spc[:,my_wl <= split_wl]
spc_high = my_spc[:,my_wl > split_wl]

my_sum_low = np.sum(spc_low, axis = 1)
my_sum_high = np.sum(spc_high, axis = 1)

select_low = my_sum_low != 0
select_high = my_sum_high != 0
select = my_sum_low * my_sum_high != 0

coord = np.arange(len(spc_low))
zeros = np.zeros(len(spc_low))

coord = coord[select]
spc_low_s = spc_low[select]
spc_high_s = spc_high[select]
my_sum_low_s = my_sum_low[select]
my_sum_high_s = my_sum_high[select]

spc_low_n = np.array([spc/s for spc, s in zip(spc_low_s, my_sum_low_s)])
spc_high_n = np.array([spc/s for spc, s in zip(spc_high_s, my_sum_high_s)])

spc_train = np.concatenate([spc_low_s, spc_high_s], axis = 1)
my_sum = np.sum(spc_train, axis = 1)
spc_norm = np.array([spc/s for spc, s in zip(spc_train, my_sum)])

#%%
# PlotIt
mean_spc = mean_spectrum(spc_norm, my_data)

#%%
# simple explort
np.savetxt(r'C:\Users\ungersebastian\Programmierung\Python\Projekte\irafmimaging\meanspc.txt', mean_spc)

#%%

from sklearn.decomposition import PCA
ncomp = 2
model = PCA(n_components=ncomp)

transformed_data = model.fit(spc_norm-mean_spc).transform(spc_norm-mean_spc).T
loadings = model.components_

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
for icomp in range(ncomp):
    maps[icomp][coord] = transformed_data[icomp]
    maps[icomp] = np.reshape(maps[icomp], (my_data['xPixel'], my_data['yPixel']) )
    
    my_fig = plt.figure()
    ax = plt.subplot(111)
    plt.colorbar( ax.imshow(maps[icomp], cmap = 'coolwarm', extent = my_data.extent()) )
    
    ax.set_xlabel('x scan ['+my_data['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data['YPhysUnit']+']')
    plt.title('factors PC'+str(icomp+1))
    

    my_fig.tight_layout()

