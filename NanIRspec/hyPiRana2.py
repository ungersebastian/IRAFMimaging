"""
---------------------------------------------------------------------------------------------------

	@author Sebastian Unger, René Lachmann
	@email herr.rene.richter@gmail.com
	@create date 2021-09-15 13:54:24
	@modify date 2021-09-16 10:36:00
	@desc [description]

---------------------------------------------------------------------------------------------------
"""

# %% imports & parameters
import os.path as path
import numpy as np
import matplotlib.pyplot as plt

from utils_chemometrics import norm
from IRAFM import IRAFM as ir

"""#####################################################"""
path_project = path.dirname(path.realpath(__file__))
path_final = path.join(path_project, 'resources', '2107_weakDataTypeJuly')
#path_final = '/home/tanoshimi/Programming/python/collaborations/TAEUBER_DANIELA/irafmimaging/NanIRspec/resources/2107_weakDataTypeJuly/'
headerfile = 'BacVan30_0013.txt'
#headerfile = 'data.npy'

"""#####################################################"""

my_data = ir(path_final, headerfile)
#my_data = np.load(path.join(path_final, headerfile))
pos = [my_file['Caption'] == 'hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = np.reshape(hyPIRFwd['data'], (hyPIRFwd['data'].shape[0] *
                  hyPIRFwd['data'].shape[1], hyPIRFwd['data'].shape[2]))
#data_train = data[my_sum != 0]

# different p-norms
l1 = np.sum(data, axis=1)
l2 = np.sqrt(np.sum(data**2, axis = 1))

# shape of the hyperspectral datacube
n_data = data.shape[0]
n_chan = data.shape[1]

dim = np.sqrt(n_data).astype(int)
dim = (dim, dim)

data_norm = np.array([d/n if n > 0 else d*0 for d,n in zip(data, l2)])
data_id = np.arange(n_data)

# remove some spectra

rm_select = l1==0
rm_select[570] = True # this one strange spc
rm_id = data_id[rm_select]

data = data[rm_select==False]
data_norm = data_norm[rm_select==False]
data_id = data_id[rm_select==False]
data_id_clean = np.arange(len(data_id))
# %% creating lens values

"""########### lens 1: 1D, Variance ###################"""

def lens1D_variance(data):
    return np.var(data, axis=1)

lens1 = lens1D_variance(data)

fig = plt.figure()
title = '1D lens value histogram'
fig.canvas.set_window_title(title)
plt.hist(lens1, bins = np.ceil(np.sqrt(n_data)).astype(int))
plt.title(title)
plt.xlabel('variance')
plt.show()

"""###### lens 2: 2D, NearestNeighbors ########"""

from sklearn.neighbors import NearestNeighbors
import networkx as nx

n_neighbors = 10
nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(data_norm)
distances, indices = nbrs.kneighbors(data_norm)
weights= 1/(distances+np.mean(distances)*(1E-5))

G = nx.Graph()
G.add_nodes_from(data_id_clean)
[[G.add_edge(node, ind, length = edge_len) for ind, edge_len in zip(node_ind, node_edge)] for node, node_ind, node_edge in zip(data_id_clean,indices,weights)]

pos = nx.spring_layout(G, weight='length')
lens2 = np.array(list(pos.values()))

fig = plt.figure()
title = '2D lens values, KNN'
fig.canvas.set_window_title(title)
nx.draw(G, pos, node_size = 1)
plt.title(title)
plt.show()

"""###### lens 3: nD, PCA ########"""

from sklearn.decomposition import PCA

var_ceil = 0.53
data_pca = data_norm - np.mean(data_norm, axis = 0)


pca = PCA(n_components=n_chan)
pca.fit(data_pca)

var = pca.explained_variance_ratio_

var_sum = []
var_old = 0
for v in var:
    var_sum.append(v+var_old)
    var_old = v+var_old

n_comp = np.where(np.asarray(var_sum)>var_ceil)[0][0]
n_comp = np.amax([3, n_comp])
lens3 = pca.transform(data_norm - np.mean(data_norm, axis = 0))[:,:n_comp]

fig = plt.figure()
title = 'nD lens values, PCA, comp1+2'
fig.canvas.set_window_title(title)
plt.scatter(lens3[:,0],lens3[:,1])
plt.title(title)
plt.show()

lens_val = [lens1, lens2, lens3]

# %% creating sub sets

lens = lens_val[1]

subset_n = 40
subset_over = 0.1
if lens.ndim == 1:
    n_lens_dim = 1
else:
    n_lens_dim = lens.shape[-1]

minmax = np.amax(lens, axis = 0) - np.amin(lens, axis = 0)
mins = np.amin(lens, axis = 0)
channels = subset_n

if n_lens_dim > 1:
    sd = 3*np.std(lens, axis = 0)
    weights = sd/minmax
    start = subset_n**(1/n_lens_dim)
    fac = (subset_n / np.prod(start*weights))**(1/n_lens_dim)
    channels = np.ceil(fac*start*weights).astype(int)
    channel_width = minmax/channels
else: 
    channel_width = [minmax/channels]
    channels = [channels]
    minmax = [minmax]

borders = [
    [ [ mm-subset_over*cw + kChan*cw,
        mm+subset_over*cw + (kChan+1)*cw] for kChan in range(c)]
    for mm, cw, c in zip(mins, channel_width,channels) ]

#%% sorting lens values in subsets

# part 1: creating an enumerated list of all possible permutations of rectangles

import itertools

permutationList = list(itertools.product(*[range(c) for c in channels]))

weights = channels.__copy__()
weights = np.flip(weights)
weights[-1]=1
weights = np.flip([np.prod(weights[0:i]) for i in range(len(weights))])

idPL = [np.sum([ w*p for w, p in zip(weights, perm) ]).astype(int) for perm in permutationList]

FunID = lambda x: np.sum(x*weights).astype(int)
FunID_inv = lambda x: permutationList[x]

# part 2: look into each rectangle and look for possible spc ids

hcidl = []

for mi in permutationList:
    a = [True,]*lens.shape[0]
    for m in range(len(channels)):
        mb=borders[m][mi[m]]
        a*=(lens[:,m]>mb[0])*(lens[:,m]<mb[1])
    hcidl.append(data_id_clean[a])
    #

#%%

from scipy.cluster.hierarchy import linkage, dendrogram
id_spc =hcidl[0]

dat = data_norm[id_spc]
x = linkage(dat)

fig = plt.figure(figsize=(25, 10))

dn = dendrogram(x)

plt.show()