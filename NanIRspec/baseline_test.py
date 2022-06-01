# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:55:47 2022

@author: basti
"""

#%% imports
# public
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.signal import medfilt
import networkx as nx
import functools
import csv
import scipy.optimize as optimize
from scipy.optimize import leastsq 
from numpy.linalg import solve
# private
from IRAFM import IRAFM as ir

"""#####################################################"""

def catshow(imlist, namelist = ['',], normalize = False, axis = 1, cmap = 'gray'):
    name = []
    for k in namelist:
        name.append(k)
        name.append(' - ')
    name = ''.join(name)
    
    if normalize:
        for ia, a in enumerate(imlist):
            amin = np.amin(a)
            amax = np.amax(a)
            a = (a-amin)/(amax-amin)
            imlist[ia] = a#/np.std(a)
    fig = np.concatenate(imlist, axis = axis)
    
    plt.figure(name)
    plt.imshow(fig, cmap = cmap)
    plt.show()
    
def ortho_poly(x, deg=3):
    x = np.asarray(x)
    n = deg + 1
    xm = np.mean(x)
    x = x - xm
    X = np.fliplr(np.vander(x, n))
    q,r = np.linalg.qr(X)
    z = np.diag(np.diag(r))
    raw = np.dot(q, z)
    norm2 = np.sum(raw**2, axis=0)
    Z = raw / np.sqrt(norm2)
    return Z.T

Z = ortho_poly(np.arange(100))

plt.figure()
for z in Z:
    plt.plot(z)
#%%

"""#####################################################"""

# loading the data
path_project = path.dirname(path.realpath(__file__))

# Einlesen der Spektren, Beschneidung des spektralen Bereiches und Kalibrierung mit CaF2

use_calibration = True
eps = 1E-10
path_resources = path.join(path_project, r'resources\BacillusSubtilis')
files = {
    #'2104_DalVan'   :   ['DalVan',  '2104_DalVan_1351-1659cm-1',    'Dalvan0006.txt',     'avCaF2_DalVan0006.txt'       ], 
    #'2105_BacVan30' :   ['BacVan',  '2105_BacVan30_1351-1659cm-1',  'BacVan30_0014.txt',  'avCaF2_forBacVan30_0014.txt' ],
    #'2107_BacVan30' :   ['BacVan',  '2107_BacVan30_1351-1659cm-1',  'BacVan30_0012.txt',  'avCaF2_BacVan30_0012.txt'    ],
    #'2107_Control30':   ['Control', '2107_Control30_1351-1659cm-1', 'Control30_0016.txt', 'avCaF2_forControl30_0016.txt'],
    #'2108_BacVan15' :   ['BacVan',  '2108_BacVan15_1400-1659cm-1',  'BacVan15_0007.txt',  'avCaF2_BacVan15_0007.txt'    ],
    '2108_BacVan30' :   ['BacVan',  '2108_BacVan30_1400-1659cm-1',  'BacVan30_0011.txt',  'avCaF2_forBacVan30.txt'      ],
    #'2108_BacVan60' :   ['BacVan',  '2108_BacVan60_1400-1659cm-1',  'BacVan60_0013.txt',  'avCaF2_BacVan60_0013.txt'    ],
    #'2108_Control30':   ['Control', '2108_Control30_1400-1659cm-1', 'Control30_0016.txt', 'avCaF2_forControl30.txt'     ],
    #'2108_Control60':   ['Control', '2108_Control60_1400-1659cm-1', 'Control60_0011.txt', 'avCaF2_Control60_0011.txt'   ]
    }

keys = list(files.keys())
k = keys[0]
p = path.join(path_resources, files[k][1], files[k][3])
with open(p, 'r') as f:
    csv_reader = csv.reader(f,delimiter='\t')
    lines = []
    for row in csv_reader:
        lines.append(row[1])
    lines = lines[1:]

data = {}
lengths = []
for k in keys:
    
    my_data = ir( 
        path.join(path_resources, files[k][1]),
        files[k][2])
   
    wl = my_data['wavelength']
    data[k] = [k[0], my_data, wl]
    
    lengths.append(len(wl))
    #print(k, ' n-wl: ', len(wl))
    
# all spc share the same wl-datapoints --> only clipping/no interpolation
# i've written a fully working calibration script if this is ever needed --> ask me 
# this method here works only if every referennce wl-datapoint is present in the current one!

amin = np.argmin(lengths)
wavelength = data[keys[amin]][2]

for k in keys:
    
    my_data = data[k][1]
    
    wTemp = data[k][2]
    wTemp = np.sum([wTemp == w for w in wavelength], axis = 0)==1
    
    # calibration file
    calib = []
    p = path.join(path_resources, files[k][1], files[k][3])
    with open(p, 'r') as f:
        csv_reader = csv.reader(f,delimiter='\t')
        
        for row in csv_reader:
            calib.append(row[1])
        calib = calib[1:]
    
    calib = np.array(calib).astype(float)[wTemp]
    
    # the spectra

    pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
    hyPIRFwd = np.array(my_data['files'])[pos][0]
    
    spc = np.reshape(hyPIRFwd['data'], (1,*hyPIRFwd['data'].shape))
    
    wTemp = np.reshape(np.repeat(wTemp , np.prod(spc.shape[:-1])), spc.shape, order = 'F')
    
    spc = np.reshape(spc[wTemp], (*spc.shape[:-1], len(wavelength)))
    
    # remove zero value spectra
    
    pos = np.sum(spc, axis = -1) == 0
    spc[pos] = spc[pos]+eps
    
    calib = np.reshape(np.repeat(calib , np.prod(spc.shape[:-1])), spc.shape, order = 'F')
    
    if use_calibration:
        #plt.figure()                              # comparisson
        #plt.plot(spc[0,0,0]/np.sum(spc[0,0,0]))   # comparisson
        spc = spc/calib
        #plt.plot(spc[0,0,0]/np.sum(spc[0,0,0]))   # comparisson
        #plt.show()                                # comparisson
    
    spc = spc + 1E-10
    
    data[k] = [k[0], spc]
    #print(k, ' new shape: ', spc.shape)


spc_data  = np.array([data[k][1][0] for k in keys])
spc_class = [data[k][0] for k in keys]
spc_keys  = keys
spc_wl    = wavelength
"""
for fig, name in zip(np.sum(spc_data, -1), spc_keys):

    plt.figure(name)
    plt.imshow(fig)
    plt.show()
"""
catshow(np.sum(spc_data, -1), ['intensity images: ',]+spc_keys, True)    


del(spc, keys, wTemp, calib, wavelength, data, amin, csv_reader, eps, f, files, hyPIRFwd, k, lengths, lines, my_data, p, pos, row, use_calibration, wl)
#if data.ndim < 4:
#    data = np.reshape(data, (*list(np.ones(4-data.ndim).astype(int)), *data.shape))
    
################

nZ,*imshape, vChan = spc_data.shape

#%%
from scipy.signal import savgol_filter as savgol 


spc = np.reshape(spc_data, (np.prod(spc_data.shape[:-1]), vChan))

tmp = spc[200]

poly = 2
wl   = 3

s0 = savgol(tmp,
            window_length = wl,
            polyorder = poly,
            deriv = 0,
            delta = 1,
            axis = 0)

s1 = savgol(tmp,
            window_length = wl,
            polyorder = poly,
            deriv = 1,
            delta = 1,
            axis = 0)

s2 = savgol(tmp,
            window_length = wl,
            polyorder = poly,
            deriv = 2,
            delta = 1,
            axis = 0)


pos = np.where(np.diff(np.sign(s1), axis = 0) > 0)[0]

iMax = pos+ s1[pos] / ( s1[pos] - s1[pos+1] ) 

P0_0, P0_1, P1_0, P1_1, P2_0, P2_1 = s0[pos], s0[pos+1], s1[pos], s1[pos+1], s2[pos], s2[pos+1]
X1_0, X1_1 = pos, (pos+1)
X2_0, X2_1 = X1_0**2, X1_1**2
X3_0, X3_1 = X1_0*X2_0, X1_1*X2_1
X4_0, X4_1 = X2_0**2, X2_1**2
X5_0, X5_1 = X4_0*X1_0, X4_1*X1_1

X0 = np.zeros(len(X1_0))

bigmat = np.array([
 [X0+1, X1_0,   X2_0,   X3_0,    X4_0,    X5_0],
 [X0+1, X1_1,   X2_1,   X3_1,    X4_1,    X5_1],
 [X0  , X0+1, 2*X1_0, 3*X2_0, 4* X3_0, 5* X4_0],
 [X0  , X0+1, 2*X1_1, 3*X2_1, 4* X3_1, 5* X4_1],
 [X0  , X0  ,   X0+2, 6*X1_0, 12*X2_0, 20*X3_0],
 [X0  , X0  ,   X0+2, 6*X1_1, 12*X2_1, 20*X3_1]
 ])

X = np.array([bigmat[:,:,i] for i in range(bigmat.shape[-1])])

bigmat = np.array([
 P0_0, P0_1, P1_0, P1_1, P2_0, P2_1
])

Y = np.array([bigmat[:,i] for i in range(bigmat.shape[-1])])

from numpy.linalg import solve

res = solve(X, Y)

yMax = np.sum(res*np.array([iMax**i for i in range(6)]).T, axis = 1)

plt.figure()
plt.plot(tmp)
plt.scatter(iMax, yMax)
plt.show()



#%%
deg = 3
Z = ortho_poly(iMax, deg=deg)

def residuals(p):
    return (yMax - np.sum(np.asarray([a*z for a,z in zip(p, Z)]), axis = 0))**2


out = optimize.leastsq(
    residuals, x0 = np.ones(deg))

out = out[0]

test = np.sum(np.asarray([o*z for o,z in zip(out, Z)]), axis = 0)

plt.figure()
plt.plot(iMax, yMax)
plt.plot(tmp)
plt.plot(iMax, test)
plt.show()


#%%

lam = 5
deg = 3

Z = ortho_poly(spc_wl, deg=deg)

iF = np.floor(iMax).astype(int)
iC = iF+1
Z_loc = Z[:,iF] + (Z[:,iC]-Z[:,iF])*(iMax-iF)

def residuals_expit(p):
    val = yMax - np.sum(np.asarray([a*z for a,z in zip(p, Z_loc)]), axis = 0)
    val_neg = val.__copy__()
    val_neg[val_neg>0]=0
    return np.abs(val) - lam * val_neg

out = optimize.leastsq(
    residuals_expit, x0 = np.ones(deg))[0]

test = np.sum(np.asarray([o*z for o,z in zip(out, Z)]), axis = 0)

plt.figure()
plt.plot(spc_wl, tmp)
plt.plot(spc_wl, test)
plt.plot(spc_wl, tmp-test)
plt.plot(spc_wl, np.zeros(len(spc_wl)))
plt.show()

#%%

def base_poly(spc, deg = 4, lam = 5, poly_sav = 4, window_sav = 5, return_filtered = False):
    deg = deg+1
        # calculating the orthonormal polygons
        
    Z = ortho_poly(spc_wl, deg=deg)
    
    n_spc = len(spc)
    
    # calculating the savgol filter
    S0 = savgol(spc,
            window_length = window_sav,
            polyorder = poly_sav,
            deriv = 0,
            delta = 1,
            axis = 1)
    
    S1 = savgol(spc,
            window_length = window_sav,
            polyorder = poly_sav,
            deriv = 1,
            delta = 1,
            axis = 1)
    S2 = savgol(spc,
            window_length = window_sav,
            polyorder = poly_sav,
            deriv = 2,
            delta = 1,
            axis = 1)
    
    # calculating the positions of local minima
    
    Pos = np.array(np.where(np.diff(np.sign(S1), axis = 1 ) > 0))
    Pos = [Pos[1][Pos[0]==i] for i in range(n_spc)]
    
    baseline = []
    
    for s0, s1, s2, pos in zip (S0, S1, S2, Pos):
        iMax = pos+ s1[pos] / ( s1[pos] - s1[pos+1] ) 
        
        # calculating the values of the local minima
        P0_0, P0_1, P1_0, P1_1, P2_0, P2_1 = s0[pos], s0[pos+1], s1[pos], s1[pos+1], s2[pos], s2[pos+1]
        X1_0, X1_1 = pos, (pos+1)
        X2_0, X2_1 = X1_0**2, X1_1**2
        X3_0, X3_1 = X1_0*X2_0, X1_1*X2_1
        X4_0, X4_1 = X2_0**2, X2_1**2
        X5_0, X5_1 = X4_0*X1_0, X4_1*X1_1
        
        X0 = np.zeros(len(X1_0))
        
        bigmat = np.array([
         [X0+1, X1_0,   X2_0,   X3_0,    X4_0,    X5_0],
         [X0+1, X1_1,   X2_1,   X3_1,    X4_1,    X5_1],
         [X0  , X0+1, 2*X1_0, 3*X2_0, 4* X3_0, 5* X4_0],
         [X0  , X0+1, 2*X1_1, 3*X2_1, 4* X3_1, 5* X4_1],
         [X0  , X0  ,   X0+2, 6*X1_0, 12*X2_0, 20*X3_0],
         [X0  , X0  ,   X0+2, 6*X1_1, 12*X2_1, 20*X3_1]
         ])
        
        X = np.array([bigmat[:,:,i] for i in range(bigmat.shape[-1])])
        
        bigmat = np.array([
         P0_0, P0_1, P1_0, P1_1, P2_0, P2_1
        ])
        
        Y = np.array([bigmat[:,i] for i in range(bigmat.shape[-1])])
        res = solve(X, Y)
        yMax = np.sum(res*np.array([iMax**i for i in range(6)]).T, axis = 1)
        
        iF = np.floor(iMax).astype(int)
        iC = iF+1
        Z_loc = Z[:,iF] + (Z[:,iC]-Z[:,iF])*(iMax-iF)
        
        # curve fitting with constraints
        
        def residuals_expit(p):
            val = yMax - np.sum(np.asarray([a*z for a,z in zip(p, Z_loc)]), axis = 0)
            val_neg = val.__copy__()
            val_neg[val_neg>0]=0
            return np.abs(val) - lam * val_neg
        
        out = optimize.leastsq(
            residuals_expit, x0 = np.ones(deg))[0]
        
        line = np.sum(np.asarray([o*z for o,z in zip(out, Z)]), axis = 0)
        
        baseline.append(line)
    
    if return_filtered:
        return S0-baseline
    else:
        return spc-baseline

#%%

import time

a = time.perf_counter()

res = base_poly(spc, deg = 20, lam = 5, poly_sav = 4, window_sav = 5, return_filtered = False)

b = time.perf_counter()
print(b-a)

iTest = 100
plt.figure()
plt.plot(spc_wl, spc[iTest])
plt.plot(spc_wl, res[iTest])
plt.plot(spc_wl, spc[iTest]-res[iTest])
plt.plot(spc_wl, np.zeros(len(spc_wl)))
plt.show()

#%%
nc = 5
lim = 3
my_spc = np.array(res)

su = np.sqrt(np.sum(my_spc**2, axis = 1))
ds = np.array([i_ds/i_s if i_s > 0 else i_ds*0 for i_ds, i_s in zip(my_spc, su)])

mean = np.mean(ds, axis = 0)
ds = ds-mean

from sklearn.decomposition import PCA
pca = PCA(n_components=nc)
fit = pca.fit(ds)

comp = fit.components_

off = np.amax(np.std(comp, axis = 0))
comp = np.array([c+i*off for i, c in enumerate(comp)])

plt.figure('with baseline correction')
for i, c in enumerate(comp):
    plt.plot(c)
    plt.plot(np.zeros(len(spc_wl))+i*off, c = 'gray')
plt.show()

sc = fit.transform(ds).T
"""
for s in sc:
    mean = np.mean(s)
    std = np.std(s)
    s[s<mean-lim*std] = mean-lim*std
    s[s>mean+lim*std] = mean+lim*std
    plt.figure()
    plt.imshow(np.reshape(s, imshape))#, cmap = 'gray')
    plt.show()

    #plt.figure()
    #plt.imshow(np.reshape(s*su, imshape), cmap = 'hsv')
    #plt.show()

"""


#%%
my_spc = np.array(spc.__copy__())

s = np.sqrt(np.sum(my_spc**2, axis = 1))
ds = np.array([i_ds/i_s if i_s > 0 else i_ds*0 for i_ds, i_s in zip(my_spc, s)])

mean = np.mean(ds, axis = 0)
ds = ds-mean

from sklearn.decomposition import PCA
pca = PCA(n_components=nc)
fit = pca.fit(ds)

comp = fit.components_
off = np.amax(np.std(comp, axis = 0))
comp = np.array([c+i*off for i, c in enumerate(comp)])

plt.figure('without baseline correction')
for i, c in enumerate(comp):
    plt.plot(c)
    plt.plot(np.zeros(len(spc_wl))+i*off, c = 'gray')
plt.show()

sc = fit.transform(ds).T
"""
for s in sc:
    mean = np.mean(s)
    std = np.std(s)
    s[s<mean-lim*std] = mean-lim*std
    s[s>mean+lim*std] = mean+lim*std
    plt.figure()
    plt.imshow(np.reshape(s, imshape), cmap = 'gray')
    plt.show()
    
plt.figure()
plt.scatter(sc[0],sc[1])
plt.show()

plt.figure()
plt.scatter(sc[0],sc[2])
plt.show()

plt.figure()
plt.scatter(sc[1],sc[2])
plt.show()
"""
"""
sc = [(a-np.amin(a))/b for a ,b in zip(sc, np.std(sc, axis = 1))]
d = squareform(pdist(sc, 'seuclidean'))
plt.figure()
plt.imshow(d)
plt.show()
"""
