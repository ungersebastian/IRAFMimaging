#%% import of the data

# imports

from os.path import isfile, join, splitext
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

path_import = r'PiFM/MCNPs/200208_Per-PImAA-NP'
headerfile = 'PeryPImAA0028.txt'

path_dir = r'//mars/usr/FA8_Mikroskopie/FAG82_BiomedizinischeBildgebung/BioPOLIM/'
path_final = join(path_dir, path_import)

data_type = np.dtype(np.int32)

def return_value(v):
    try:
        v = int(v)
    except:
        try:
            v = float(v)
        except:
            pass
    return v

def return_dict(arr):
    arr = arr[[':' in l for l in arr]]
    arr = [''.join(l.split()) for l in arr]
    arr = [''.join(l.split('\n')) for l in arr]
    return {l.split(':', 1)[0]: return_value(l.split(':', 1)[1]) for l in arr}

file_list = np.array([(
        f,
        join(path_final, f),
        splitext(f)[1]
        ) for f in listdir(path_final) if isfile(join(path_final, f))])

name = splitext(headerfile)[0]

path_file = file_list[file_list[:,0]==headerfile][0,1]

with open(path_file, 'r') as fopen:
     header_list = np.array(fopen.readlines())

# extract the file list and supporting information

where = np.where(header_list == 'FileDescBegin\n')[0]
where = np.append(where, len(header_list))
files = [header_list[where[i]:where[i+1]] for i in range(len(where)-1)]
files[-1] = files[-1][0:np.where(files[-1] == 'FileDescEnd\n')[0][0]+1]
files = [f[(np.where(f == 'FileDescBegin\n')[0][0]+1):(np.where(f == 'FileDescEnd\n')[0][0]-1)] for f in files]

header_list = header_list[0:where[0]-1]

del(where)

my_data = return_dict(header_list)

del(header_list)

# get the wavelength axis
path_wavelengths = my_data['FileNameWavelengths']
path_wavelengths = file_list[file_list[:,0]==path_wavelengths][0,1]

with open(path_wavelengths, 'r') as fopen:
     wavelength = fopen.readlines()

wavelength = [''.join(l.split('\n')) for l in wavelength][1:]
wavelength = (np.array([l.split('\t') for l in wavelength]).T).astype(float)
my_data['wavelength'], my_data['attenuation'] = wavelength[0], wavelength[1]

del(wavelength, path_wavelengths)

# extract the file information

newfile ={
 'FileName' : my_data.pop('FileName'),
 'Caption' : my_data.pop('Caption'),
 'Scale' : my_data.pop('Scale'),
 'PhysUnit' : my_data.pop('PhysUnit')
 }

files = [return_dict(f) for f in files]
files.append(newfile)

my_data['files'] = files

del(files, newfile)

# read the images

for my_file in my_data['files']:
    path_file = join(path_final, my_file['FileName'])
    with open(path_file, 'rb') as fopen:
        my_im = fopen.read()
        my_im = np.frombuffer(my_im, data_type)
        my_dim = int(len(my_im)/(my_data['xPixel']*my_data['yPixel']))
        if my_dim == 1:
            news = (my_data['xPixel'], my_data['yPixel'])
        else:
            news = (my_data['xPixel'], my_data['yPixel'], my_dim)
        my_im = np.reshape(my_im, news)
        
    scale = my_file['Scale']
    my_file['data'] = my_im*scale

#%%
    
dpx = my_data['XScanRange']/my_data['xPixel']
xlim = (my_data['xCenter'] - dpx*(my_data['xPixel']-1)/2, my_data['xCenter'] + dpx*(my_data['xPixel']-1)/2)

dpy = my_data['YScanRange']/my_data['yPixel']
ylim = (my_data['yCenter'] - dpx*(my_data['yPixel']-1)/2, my_data['yCenter'] + dpy*(my_data['yPixel']-1)/2)

extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

for my_file in my_data['files']:
    
    my_fig = plt.figure()
    ax = plt.subplot(111)
    
    data = my_file['data']
    if data.ndim > 2:
        plt.colorbar( ax.imshow(np.sum(data, axis = 2), extent = extent), label = my_file['PhysUnit'] )
    else:
        plt.colorbar( ax.imshow(data, extent = extent), label = my_file['PhysUnit'])
    
    ax.set_xlabel('x scan ['+my_data['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data['YPhysUnit']+']')
    plt.title(my_file['Caption'])
    

    my_fig.tight_layout()

pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = np.reshape(hyPIRFwd['data'], (hyPIRFwd['data'].shape[0]*hyPIRFwd['data'].shape[1], hyPIRFwd['data'].shape[2]))
my_sum = np.sum(data, axis = 1)

coord = np.arange(len(my_sum))
zeros = np.zeros(len(my_sum))
data = data[my_sum != 0]
coord = coord[my_sum != 0]
my_sum = my_sum[my_sum != 0]


data_norm = np.array([spc/s for spc, s in zip(data, my_sum)])

mean_spc = np.mean(data_norm, axis = 0)
std_spc = np.std(data_norm, axis = 0)

my_fig = plt.figure()
ax = plt.subplot(111)
ax.plot(my_data['wavelength'], mean_spc)
ax.fill_between(x = my_data['wavelength'], y1 = mean_spc+std_spc, y2 = mean_spc-std_spc, alpha = 0.2)
ax.set_xlabel('wavenumber ['+my_data['PhysUnitWavelengths']+']')
ax.set_ylabel('intensity (normalized)')
ax.set_yticklabels([])
plt.title('mean spectrum')

my_fig.tight_layout()

from sklearn.decomposition import PCA
ncomp = 2
model = PCA(n_components=ncomp)

transformed_data = model.fit(data_norm-mean_spc).transform(data_norm-mean_spc).T
loadings = model.components_

my_fig = plt.figure()
ax = plt.subplot(111)
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
    plt.colorbar( ax.imshow(maps[icomp], cmap = 'coolwarm', extent = extent) )
    
    ax.set_xlabel('x scan ['+my_data['XPhysUnit']+']')
    ax.set_ylabel('y scan ['+my_data['YPhysUnit']+']')
    plt.title('factors PC'+str(icomp+1))
    

    my_fig.tight_layout()
