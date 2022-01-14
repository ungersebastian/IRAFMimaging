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

data = np.load(r'N:\Daten_Promotions_Sebastian\raman.npy')#[10:-40,20:-30,200:400]
shape = data.shape
plt.figure()
plt.imshow(np.sum(data, axis = -1))
plt.show()

data = np.reshape(data, (np.prod(shape[:2]), shape[2]))
               
# shape of the hyperspectral datacube
n_data = data.shape[0]
n_chan = data.shape[1]

dim = np.sqrt(n_data).astype(int)
dim = (dim, dim)

data_id = np.arange(n_data)

# remove some spectra

data_id_clean=data_id

# part 1: norm

data_norm = norm(data,'std', centered = True)

# part 2: choose metric

import scipy

dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data_norm))

lens = np.log(np.sum(np.exp(-dist**2), axis = 1))

plt.figure()
plt.hist(lens)
plt.show()

im = np.reshape(lens, (shape[0], shape[1]))

plt.figure()
plt.imshow(im, cmap = 'rainbow')
plt.show()

im2 = im.__copy__().flatten()
off = 0.01
lim = np.quantile(lens,(0+off,1-off))
im2[im2<lim[0]]=lim[0]
im2[im2>lim[1]]=lim[1]
im2 = (im2-np.amin(im2))/(np.amax(im2)-np.amin(im2))
im2 = np.reshape(im2, (shape[0], shape[1]))


plt.figure()
plt.imshow(im2, cmap = 'hsv')
plt.show()

plt.figure()
plt.hist(im2.flatten())
plt.show()

scale = 1E3
img_list = list((im2.__copy__().flatten()*scale).astype(int))
histogram_array = np.bincount(img_list, minlength=shape[0])
num_pixels = np.sum(histogram_array)
histogram_array = histogram_array/num_pixels

chistogram_array = np.cumsum(histogram_array)

transform_map = np.floor(scale * chistogram_array)

eq_img_list = [transform_map[p] for p in img_list]

eq_img_array = np.reshape(np.asarray(eq_img_list), shape[:2])

plt.figure()
plt.imshow(eq_img_array, cmap = 'hsv')
plt.show()

plt.figure()
plt.hist(eq_img_array.flatten())
plt.show()


#%%

from sklearn.decomposition import PCA
pca = PCA(n_components=n_chan)
pca.fit(data_norm)

lens3 = pca.transform(data_norm - np.mean(data_norm, axis = 0))[:,1]

#%%
# creating sub sets
#my_lens = eq_img_array.flatten()
my_lens = lens3
#my_lens = im2.flatten()
method = 'distance'
#method = 'maxclust'
maxclust = 2
subset_n = 60
subset_over = 0.7

tFix = np.mean(dist)*0.25
#tFix = 1.5
w = 0

if my_lens.ndim == 1:
    n_lens_dim = 1
else:
    n_lens_dim = my_lens.shape[-1]

minmax = np.amax(my_lens, axis = 0) - np.amin(my_lens, axis = 0)
mins = np.amin(my_lens, axis = 0)
channels = subset_n

if n_lens_dim > 1:
    sd = 3*np.std(my_lens, axis = 0)
    weights = sd/minmax
    start = subset_n**(1/n_lens_dim)
    fac = (subset_n / np.prod(start*weights))**(1/n_lens_dim)
    channels = np.ceil(fac*start*weights).astype(int)
    channel_width = minmax/channels
else: 
    channel_width = [minmax/channels]
    channels = [channels]
    minmax = [minmax]
    mins = [mins]
    my_lens = np.reshape(my_lens, (my_lens.shape[0],1))
    
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
import networkx as nx

topo_graph = nx.Graph()
i_node = 0

for mi in permutationList:
    a = [True,]*my_lens.shape[0]
    
    for m in range(len(channels)):
        mb=borders[m][mi[m]]
        
        a*=(my_lens[:,m]>mb[0])*(my_lens[:,m]<mb[1])
    if np.sum(a) == 1:
        spc_id = data_id_clean[a]
        topo_graph.add_node(i_node, ids = spc_id, height = len(spc_id))
        i_node = i_node+1
    elif np.sum(a) > 1:
    
        spc_id = data_id_clean[a]
       
        # perform single linkage clustering
        ids = data_id_clean[a]
        idn = len(ids)
        ids = np.meshgrid(ids, ids)
        ida, idb = ids[0].flatten().astype(int), ids[1].flatten().astype(int)
        idd = scipy.spatial.distance.squareform(np.reshape(dist[ida,idb], (idn,idn)))
        
        Z = linkage(idd, 'single')
        
        if method == 'maxclust':
            cluster = fcluster(Z, t = np.sum(a)//maxclust, criterion = 'maxclust')
            #cluster = fcluster(Z, t = maxclust, criterion = 'maxclust')
        elif  method == 'distance':
            cluster = fcluster(Z, t = (w*t*max(Z[:,2])+(1-w)*tFix), criterion='distance')
            #cluster = fcluster(Z, t = tFix*subset_n/np.prod(mi), criterion='distance')
        n_cluster = len(np.unique(cluster))
                
        print(mi, np.sum(a), n_cluster,max(Z[:,2]))
        for i_c in range(n_cluster):
            id_list = spc_id[cluster==i_c+1]
            if len(id_list) > 1: #more than 1 spc per cluster?
                topo_graph.add_node(i_node, ids = id_list, height = len(id_list))
                i_node = i_node+1
           
    
        

n_nodes = len(topo_graph)
rm = []
for node_a in range(n_nodes):
    delete = 1
    l1 = topo_graph.nodes[node_a]['ids']
    for node_b in range(node_a+1, n_nodes):
        l2 = topo_graph.nodes[node_b]['ids']
        d = set(l1).intersection(l2)
        if (len(d)>0):
            topo_graph.add_edge(node_a, node_b, weight=len(d)) 
            delete = 0
    if delete == 1:
        rm = rm+[node_a]
rm = np.flip(rm)
for r in rm:
    topo_graph.remove_node(r)
    
pos_topo = nx.spring_layout(topo_graph, weight='weight')
fig = plt.figure()
title = 'TOPO'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo, node_size = 1)
#plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = np.log(list(nx.get_node_attributes(topo_graph, 'height').values()))+1)
plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(nx.get_node_attributes(topo_graph, 'height').values()))
plt.title(title)
plt.show()

pos_topo2 = nx.spring_layout(topo_graph)
fig = plt.figure()
title = 'TOPO'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo2, node_size = 1)
#plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = np.log(list(nx.get_node_attributes(topo_graph, 'height').values()))+1)
plt.scatter(*np.transpose(np.array(list(pos_topo2.values()))), c = list(nx.get_node_attributes(topo_graph, 'height').values()))
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