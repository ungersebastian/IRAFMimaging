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
#data_id_clean=data_id
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



#method = 'maxclust'
maxclust = 3

method = 'distance'

subset_n = 18
subset_over = 0.5
t = 0.6

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

# sorting lens values in subsets

# part 1: creating an enumerated list of all possible permutations of rectangles

import itertools

permutationList = list(itertools.product(*[range(c) for c in channels]))

# part 2: look into each rectangle and look for possible spc ids

from scipy.cluster.hierarchy import linkage, fcluster,dendrogram

topo_graph = nx.Graph()
i_node = 0

for mi in permutationList:
    a = [True,]*lens.shape[0]
    for m in range(len(channels)):
        mb=borders[m][mi[m]]
        
        a*=(lens[:,m]>mb[0])*(lens[:,m]<mb[1])
    if np.sum(a) == 1:
        spc_id = data_id_clean[a]
        topo_graph.add_node(i_node, ids = spc_id, height = len(spc_id))
        i_node = i_node+1
    elif np.sum(a) > 1:
    
        spc_id = data_id_clean[a]
       
        # perform single linkage clustering
        dat = [lens[ids] for ids in spc_id]
        Z = linkage(dat)
        #dat = data_norm[spc_id]
        if method == 'maxclust':
            cluster = fcluster(Z, t = maxclust, criterion = 'maxclust')
        elif  method == 'distance':
            cluster = fcluster(Z, t = t*max(Z[:,2]), criterion='distance')
        
        """
        plt.figure()
        plt.scatter(*np.transpose(dat), c = cluster)
        plt.show()
        """
        n_cluster = len(np.unique(cluster))
        
        print(np.sum(a), len(spc_id), n_cluster)
        for i_c in range(n_cluster):
            id_list = spc_id[cluster==i_c+1]
            if len(id_list) > 0:
                topo_graph.add_node(i_node, ids = id_list, height = len(id_list))
                i_node = i_node+1
        

n_nodes = len(topo_graph)

for node_a in range(n_nodes):
    l1 = topo_graph.nodes[node_a]['ids']
    for node_b in range(node_a+1, n_nodes):
        l2 = topo_graph.nodes[node_b]['ids']
        d = set(l1).intersection(l2)
        if (len(d)>0):
            topo_graph.add_edge(node_a, node_b, weight=len(d)) 
    #%%    
pos_topo = nx.spring_layout(topo_graph, weight='weight')
fig = plt.figure()
title = 'TOPO'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo, node_size = 1)
#plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = np.log(list(nx.get_node_attributes(topo_graph, 'height').values()))+1)
plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(nx.get_node_attributes(topo_graph, 'height').values()))
plt.title(title)
plt.show()
#%%

topo_clust = nx.clustering(topo_graph, weight = 'height')
fig = plt.figure()
title = 'TOPO2'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo, node_size = 1)
#plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = np.log(list(nx.get_node_attributes(topo_graph, 'height').values()))+1)
plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(topo_clust.values()))
plt.title(title)
plt.show()

c=np.log(1+np.array(list(nx.get_node_attributes(topo_graph, 'height').values()))*np.array(list(topo_clust.values())))
fig = plt.figure()
title = 'TOPO3'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo, node_size = 1)
#plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = np.log(list(nx.get_node_attributes(topo_graph, 'height').values()))+1)
plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = c)
plt.title(title)
plt.show()

#%%

from scipy.interpolate import griddata

resolution = 1000
scale_p = np.transpose(np.array(list(pos_topo.values())).__copy__())
scale_p = np.transpose(np.array([(sp - np.amin(sp))/(np.amax(sp)-np.amin(sp)) * (resolution-1) for sp in scale_p]))

x = np.arange(resolution)
x,y = np.meshgrid(x,x)

x = np.array([[iy, ix] for ix, iy in zip(x.flatten(),y.flatten())])

Ztri = np.nan_to_num(griddata( scale_p, c,  x , method='linear').reshape((resolution,resolution)))

plt.figure()
plt.imshow(Ztri.T, extent=(0,1,0,1))
plt.show()

#%%

def func(x, y):

    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]


#rng = np.random.default_rng()

points = np.random.rand(1000, 2)

values = func(points[:,0], points[:,1])


from scipy.interpolate import griddata

grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')

grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')

grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
#%%

resolution = 512 
scale_p = np.transpose(np.array(list(pos_topo.values())).__copy__())
scale_p = np.transpose(np.round([(sp - np.amin(sp))/(np.amax(sp)-np.amin(sp)) * (resolution-1) for sp in scale_p]).astype(int))
img = np.zeros((resolution,resolution))

for i, p in enumerate(scale_p):
    img[p[0],p[1]] = c[i]
    

g = 1
gauss = lambda x: np.exp(-(x-resolution//2)**2/(2*g**2))

x = np.arange(resolution)
x,y = np.meshgrid(x,x)
x = gauss(x)*gauss(y)

blur = np.real(np.fft.ifft2(np.fft.fft2(img)*np.fft.fft2(np.fft.fftshift(x))))
blur = np.abs(blur)
mask = img.__copy__()
mask[mask>0]=1

mask = np.real(np.fft.ifft2(np.fft.fft2(mask)*np.fft.fft2(np.fft.fftshift(x))))
mask[mask<0]=0
mask=mask + np.amin(blur)+1E-3



plt.figure()
plt.imshow(blur/mask)
plt.show()