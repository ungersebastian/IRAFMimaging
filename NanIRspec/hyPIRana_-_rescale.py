# -*- coding: utf-8 -*-
"""
hyPIRana is a program for analysis of PiFM hyPIR spectra acquired using VistaScan
Created on Fri Apr 24 08:06:26 2020

@author: ungersebastian

(Last) Modified on Thu Apr 29 by Daniela Taeuber
"""

#%% imports & parameters

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from IRAFM import IRAFM
from AreaSelect import getArea
import pandas as pd

path_import = r'F:\daniela\retina\NanIRspec\resources'
headerfile = 'Ret29r20006.txt'
path_dir = r'retina/NanIRspec/resources'
pathsub="F:/daniela/retina/NanIRspec/resources/CA20001mean1350-1750.txt"
path_final = join(path_dir, path_import)
#in below lines path can be imported by panda in more general form ,if there is new text file which is not in standard form
# PATH_substrate = "CA20001mean1350-1750.txt"  # csv or txt format enter substrate data
# df_sub = pd.read_csv( PATH_substrate, sep=".",skiprows=[],delimiter="\t" )  # in skiprows write the rows which contain text in order to elliminate them
# y_sub = df_sub.values.T
#%% functions

def mean_spectrum(spc_norm, my_wl, my_data):
    '''
    Calculates and plots mean spectra with standard deviations
    '''
    mean_spc = np.mean(spc_norm, axis = 0)
    std_spc = np.std(spc_norm, axis = 0)

    my_fig = plt.figure()
    ax = plt.subplot(111)
    ax.fill_between(x = my_wl, y1 = mean_spc+std_spc, y2 = mean_spc-std_spc, alpha = 0.6)
    plt.gca().invert_xaxis() #inverts values of x-axis
    ax.plot(my_wl, mean_spc)
    ax.set_xlabel('wavenumber ['+my_data['PhysUnitWavelengths']+']')
    ax.set_ylabel('PiFM [V]') #+my_data['files'=6,'PhysUnit']+'
    #ax.set_yticklabels([])
    plt.title('mean spectrum')
    my_fig.tight_layout()
    return mean_spc , std_spc

def export_spectrum(spectrum):
    '''
    Exports (saves) spectral data in ascii(?) text(?) format  - planned
    '''
  
def PCA_spectrum(my_spectra, my_wl, my_data, my_coord):
    '''
    Calculates and plots two principal components for a set of spectral data
    '''
    from sklearn.decomposition import PCA
    ncomp = 2
    model = PCA(n_components=ncomp)
    transformed_data = model.fit(my_spectra).transform(my_spectra).T
    loadings = model.components_
    my_fig = plt.figure()
    ax = plt.subplot(111)
    plt.gca().invert_xaxis() #inverts values of x-axis
    for icomp in range(ncomp):
        ax.plot(my_wl, loadings[icomp], label='PC'+str(icomp+1) )
    ax.set_xlabel('wavenumber ['+my_data['PhysUnitWavelengths']+']')
    ax.set_ylabel('PiFM [V]')
    #ax.set_yticklabels([])
    ax.legend()
    plt.title('PCA-Loadings (principal components)')

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
        maps[icomp][my_coord] = transformed_data[icomp]
        maps[icomp] = np.reshape(maps[icomp], (my_data['xPixel'], my_data['yPixel']) )
    
        my_fig = plt.figure()
        ax = plt.subplot(111)
        plt.colorbar( ax.imshow(maps[icomp], cmap = 'coolwarm', extent = my_data.extent()) )
    
        ax.set_xlabel('x scan ['+my_data['XPhysUnit']+']')
        ax.set_ylabel('y scan ['+my_data['YPhysUnit']+']')
        plt.title('factors PC'+str(icomp+1))    

        my_fig.tight_layout() 
    return loadings , transformed_data[0], transformed_data[1]
#%% loads data and plots associated VistaScan parameter images

my_data = IRAFM(path_final, headerfile)

my_data.plot_all()

#%% Part1

testIm = my_data['files'][0]
data = testIm['data']

my_p, my_a = getArea(image = data, delta = 3)

pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = hyPIRFwd['data']
data_select = data[my_a == 1]   # here are now just the selected spectra


our_data=data[:,:,1]*np.array(my_a)

#%% checks validity of data and sorts them
w,h=my_a.shape
my_spc = my_data.return_spc()*np.reshape(my_a,newshape=(w*h,1))
#print(my_spc.shape)
#print(my_data.return_spc().shape)
#print(my_a.shape)
# my_spc = our_data
my_wl  = my_data['wavelength']
split_wl = 1644
my_wl_low = my_wl[my_wl <= split_wl]
my_wl_high = my_wl[my_wl > split_wl]
print('*'*25)
print()
spc_low = my_spc[:,my_wl <= split_wl]
spc_high = my_spc[:,my_wl > split_wl]

my_sum_low = np.sum(spc_low, axis = 1)
my_sum_high = np.sum(spc_high, axis = 1)

select_low = my_sum_low != 0
select_high = my_sum_high != 0
select = my_sum_low * my_sum_high != 0

coord = np.arange(len(spc_low))
zeros = np.zeros(len(spc_low))

#%% Evaluates spectra from spatial positions where both laser tuners recorded and plots mean spectra and PCA

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
#Rescalling
with open( pathsub, 'r' ) as fopen:
    y_sub = np.array(fopen.readlines())
y_sub = [''.join(l.split('\n')) for l in y_sub][1:]
y_sub = (np.array([l.split('\t') for l in y_sub]).T).astype(float)
y_sub=y_sub[1,:]
b=len(spc_norm)
for i in range( b ):
    spc_norm[i, :] = spc_norm[i, :] / y_sub
#rescaling finished
mean_spc = np.mean(spc_norm, axis = 0)
std_spc = np.std(spc_norm, axis = 0)
print("my_w1",my_wl.shape)
print("meanspc",mean_spc.shape)
print("std_spc",std_spc.shape)
save_mean=np.zeros((406,2),dtype=float)
save_mean[:,0]=my_wl
save_mean[:,1]=mean_spc
save_std=np.zeros((406,2),dtype=float)
save_std[:,0]=my_wl
save_std[:,1]=std_spc
np.savetxt("mean_spc.txt",save_mean,delimiter='\t',header='wavenumber\tmean_spc',fmt="%8.5f")
np.savetxt("std_spc.txt",save_std,delimiter='\t',header='wavenumber\tstd_spc',fmt="%8.5f")
join(path_final,"mean_spc.txt")
join(path_final,"std_spc.txt")
# PlotIt
mean_spc = mean_spectrum(spc_train, my_data['wavelength'], my_data)
PCA_spectrum(spc_train, my_data['wavelength'], my_data, coord) #why is this already showing normalized plots?
mean_spc = mean_spectrum(spc_norm, my_data['wavelength'], my_data)
#spc_norm-mean_spc; substracting the mean_spc does not make any difference
PCA_spectrum(spc_norm, my_data['wavelength'], my_data, coord)
comp,pc1,pc2=PCA_spectrum(spc_train, my_data['wavelength'], my_data, coord)
np.savetxt("PCA.txt",comp,delimiter='\t',header='comp',fmt="%8.5f")
np.savetxt("PC1.txt",pc1,delimiter='\t',header='pc1',fmt="%8.5f")
np.savetxt("PC2.txt",pc2,delimiter='\t',header='pc2',fmt="%8.5f")
#%% Evaluates spectra from spatial positions where the lower spectral range tuners recorded and plots mean and PCS
coord_low = np.arange(len(spc_low))
zeros = np.zeros(len(spc_low))
coord_low = coord_low[select_low]
spc_low_s = spc_low[select_low]
my_sum_low_s = my_sum_low[select_low]


spc_low_n = np.array([spc/s for spc, s in zip(spc_low_s, my_sum_low_s)])
my_sum = np.sum(spc_low_n, axis = 1)
spc_norm = np.array([spc/s for spc, s in zip(spc_low_s, my_sum)])

# PlotIt
mean_spc = mean_spectrum(spc_low_s, my_wl_low, my_data)
PCA_spectrum(spc_low_s, my_wl_low, my_data, coord_low)

mean_spc = mean_spectrum(spc_low_n, my_wl_low, my_data)
PCA_spectrum(spc_low_n, my_wl_low, my_data, coord_low)

#%% Evaluates spectra from spatial positions where the higher spectral range tuners recorded and plots mean and PCS
coord_high = np.arange(len(spc_high))
zeros = np.zeros(len(spc_high))
coord_high = coord_high[select_high]
spc_high_s = spc_high[select_high]
my_sum_high_s = my_sum_high[select_high]


spc_high_n = np.array([spc/s for spc, s in zip(spc_high_s, my_sum_high_s)])
my_sum = np.sum(spc_high_n, axis = 1)
spc_norm = np.array([spc/s for spc, s in zip(spc_high_s, my_sum)])

# PlotIt
mean_spc = mean_spectrum(spc_high_s, my_wl_high, my_data)
PCA_spectrum(spc_high_s, my_wl_high, my_data, coord_high)

mean_spc = mean_spectrum(spc_high_n, my_wl_high, my_data)
PCA_spectrum(spc_high_n, my_wl_high, my_data, coord_high)
plt.show()


