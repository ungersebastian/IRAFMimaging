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
from skimage import exposure
from utils_chemometrics import norm
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
import networkx as nx


"""#####################################################"""
path_project = path.dirname(path.realpath(__file__))

data = np.load(r'N:\Daten_Promotions_Sebastian\raman.npy')[:,:,200:400]
*imshape, veclen = data.shape

fig = plt.figure()
fig.canvas.set_window_title('original data')
plt.imshow(np.sum(data, axis = -1))
plt.show()

"""####### downscaling for faster processing ###########"""

windowwidth = 3
    
npx = [(s//windowwidth)*windowwidth for s in imshape]
dnpx = [(s-p)//2 for s, p in zip(imshape, npx)]
nwin = [p//windowwidth for p in npx]
data_red = data.__copy__()[tuple([slice(d,d+p) for d, p in zip(dnpx, npx)])]
data_red = np.median(data_red[:nwin[0]*windowwidth, :nwin[1]*windowwidth].reshape(nwin[0], windowwidth, nwin[1], windowwidth, veclen), axis = (1,3))

intIm = np.sum(data_red, axis = -1)

fig = plt.figure()
fig.canvas.set_window_title('downscaled data')
plt.imshow(intIm)
plt.show()

data = data_red
*imshape, veclen = data.shape

"""####### preprocessing ################################"""

n_data = np.prod(imshape)
data = np.reshape(data, (n_data, veclen))
data_id = np.arange(n_data)

# remove some spectra

# ( not done here )
data_id_clean=data_id

# part 1: norm

data_norm = norm(data,1, centered = True)

#%% part 2: choose metric and generate lens values

from scipy.spatial.distance import squareform, pdist

dist = squareform(pdist(data_norm))

lens_gauss_density = np.sum(np.exp(-dist**2), axis = 0)

im = np.reshape(lens_gauss_density, (imshape))
im = (im-np.amin(im))/(np.amax(im)-np.amin(im))

fig = plt.figure()
fig.canvas.set_window_title('lens: gauss density')
plt.imshow(im, cmap = 'rainbow')
plt.show()

im = exposure.equalize_hist(im)

fig = plt.figure()
fig.canvas.set_window_title('lens: gauss density, hist.-equ.')
plt.imshow(im, cmap = 'rainbow')
plt.show()

lens_gauss_density_hist = im.flatten()

pca = PCA(n_components=2)
pca.fit(data_norm)

lens_pca = pca.transform(data_norm - np.mean(data_norm, axis = 0))[:,1]
lens_pca = (lens_pca - np.amin(lens_pca))/(np.amax(lens_pca)-np.amin(lens_pca))

im = np.reshape(lens_pca, (imshape))

fig = plt.figure()
fig.canvas.set_window_title('lens: PC1')
plt.imshow(im, cmap = 'rainbow')
plt.show()

im = exposure.equalize_hist(im)

fig = plt.figure()
fig.canvas.set_window_title('lens: PC1, hist.-equ.')
plt.imshow(im, cmap = 'rainbow')
plt.show()

lens_pca_hist = im.flatten()

lens_pca_gauss = (2*lens_pca_hist-1) * (2*lens_gauss_density_hist-1)
lens_pca_gauss = (lens_pca_gauss-np.amin(lens_pca_gauss))/(np.amax(lens_pca_gauss)-np.amin(lens_pca_gauss))
lens_pca_gauss = exposure.equalize_hist(lens_pca_gauss)

fig = plt.figure()
fig.canvas.set_window_title('lens: PC1+gauss, hist.-equ.')
plt.imshow(np.reshape(lens_pca_gauss, (imshape)), cmap = 'rainbow')
plt.show()

#%%
# creating sub sets (only 1D lenses from now on)
"""
resolution = 60
gain = 7

t = 0.6
tFix = np.mean(dist)*0.6
w = 1
"""

resolution = 70
gain = 8

t = 0.6
tFix = np.mean(dist)*0.6
w = 1

eps = 1E-3 # stretch the borders to get every point



overlap = (gain-1)/gain
my_lens = lens_pca_gauss

minmax = np.amax(my_lens, axis = 0) - np.amin(my_lens, axis = 0)
eps = (minmax/resolution)*eps
mins = np.amin(my_lens, axis = 0)

c_width = (minmax+2*eps)/(1+(resolution-1)*(1-overlap))
c_dist  = c_width*(1-overlap)

p1 = mins - eps + (resolution-1)*c_dist
p2 = mins - eps + (resolution-1)*c_dist + c_width

subset_borders = [[mins - eps + r*c_dist, mins - eps + r*c_dist + c_width] for r in range(resolution)]

# part 2: look into each subset and look for possible spc ids



topo_graph = nx.Graph()
i_node = 0

for sb in subset_borders:
    a=(my_lens>=sb[0])*(my_lens<=sb[1])
    
    if np.sum(a) == 1:
        spc_id = data_id_clean[a]
        topo_graph.add_node(i_node, ids = spc_id, height = len(spc_id))
        i_node = i_node+1
    elif np.sum(a) > 1:
        spc_id = data_id_clean[a]
        
        ## perform single linkage clustering
        # select correct values from distance matrix
        ids = data_id_clean[a]
        idn = len(ids)
        ids = np.meshgrid(ids, ids)
        ida, idb = ids[0].flatten().astype(int), ids[1].flatten().astype(int)
        idd = squareform(np.reshape(dist[ida,idb], (idn,idn)))
        
        Z = linkage(idd, 'single')
        cluster = fcluster(Z, t = (w*t*max(Z[:,2])+(1-w)*tFix), criterion='distance')
        n_cluster = len(np.unique(cluster))
        
        for i_c in range(n_cluster):
            id_list = spc_id[cluster==i_c+1]
            if len(id_list) > 0: #more than 1 spc per cluster?
                topo_graph.add_node(i_node, ids = id_list, height = len(id_list))
                i_node = i_node+1
        print(np.sum(a), n_cluster)
        
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
        
n_rm = len(rm)
rm = np.flip(rm)
for r in rm:
    topo_graph.remove_node(r)
print('n_nodes: ', len(topo_graph), " = ", n_nodes , ' - ', n_rm, ' (', n_nodes-n_rm, ') ')

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
pos_topo = nx.spring_layout(topo_graph, weight = None)
fig = plt.figure()
title = 'TOPO - unweighted'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo, node_size = 1)
plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(nx.get_node_attributes(topo_graph, 'height').values()))
plt.title(title)
plt.show()

pos_topo = nx.spring_layout(topo_graph, weight='weight')
fig = plt.figure()
title = 'TOPO - weighted'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo, node_size = 1)
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

