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
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import sklearn
"""#####################################################"""
path_project = path.dirname(path.realpath(__file__))

data = np.load(r'N:\Daten_Promotions_Sebastian\raman3D.npy')[:,:,:,200:350]

nZ,*imshape, vChan = data.shape


fig = plt.figure()
fig.canvas.set_window_title('original data')
plt.imshow(np.sum(data[nZ//2], axis = -1))
plt.show()

# downscaling

def poolNd(im, windowwidth = 3, fun = np.median):
    if not isinstance(windowwidth, (list, tuple, np.ndarray)):
        ww = windowwidth*im.ndim
    elif len(windowwidth) == im.ndim:
        ww = windowwidth
    else:
        'windowwidth not valid'
        return False

    shape = data.shape
    npx = [(s//w)*w for s, w in zip(shape, ww)]
    dnpx = [(s-p)//2 for s, p in zip(shape, npx)]
    nwin = [p//w for p, w in zip(npx, ww)]
    data_red = im.__copy__()[tuple([slice(d,d+p) for d, p in zip(dnpx, npx)])]
    
    reshape = nwin.copy()
    for index, item in enumerate(ww):
            insert_index = index*2 + 1
            reshape.insert(insert_index, item)
    data_red = fun(data_red.reshape(*reshape), axis = np.arange(1,1+len(ww)*2,2))
    
    return data_red

ww = [6,15,15,1]
data_red = poolNd(data, ww, np.median)

*imshape_red, vChan_red = data_red.shape

fig = plt.figure()
fig.canvas.set_window_title('original data')
plt.imshow(np.sum(data_red[imshape_red[0]//2], axis = -1))
plt.show()

data_red = np.reshape(data_red, (np.prod(data_red.shape[:-1]),data_red.shape[-1]))
data_red_id = np.arange(len(data_red))

#%%
"""####### creating lens values #########################"""

metric_name = 'cosine' # euclidean, seuclidean, correlation

dist = squareform(pdist(data_red, metric_name))

dist_lens = dist.__copy__()
np.fill_diagonal(dist_lens, val = np.inf)

id_list = np.arange(dist_lens.shape[0])
lens_list = []

def argmatmin(mat):
    return list(np.sort(np.unravel_index(mat.argmin(), mat.shape)))

i_now, i_next = argmatmin(dist_lens)

while(len(dist_lens>0)):
    
    lens_list.append(id_list[i_now])
    i_next = argmatmin(dist_lens[i_now])[0]
    
    id_list = np.delete(id_list, i_now)
    dist_lens = np.delete(np.delete(dist_lens, i_now, axis = 0), i_now, axis = 1)
    
    i_now = i_next-1

lens = lens_list
lens = (lens-np.amin(lens))/(np.amax(lens)-np.amin(lens))

lens_im = np.reshape(lens, imshape_red[1:])

plt.figure()
plt.imshow(lens_im)

#%%
metric = lambda x1, x2: pdist([x1, x2], metric_name)
n_neighbors = 2
nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', metric = metric).fit(data_red)
distances, indices = nbrs.kneighbors(data_red)
weights= 1/(distances+np.mean(distances)*(1E-5))

G = nx.Graph()
G.add_nodes_from(data_red_id)
[[G.add_edge(node, ind, length = edge_len) for ind, edge_len in zip(node_ind, node_edge)] for node, node_ind, node_edge in zip(data_red_id,indices,weights)]

pos = nx.spring_layout(G, weight='length')
lens_red = np.array(list(pos.values()))

fig = plt.figure()
title = '2D lens values, KNN'
fig.canvas.set_window_title(title)
nx.draw(G, pos, node_size = 1)
plt.title(title)
plt.show()

lens = lens_red[:,0]*lens_red[:,1]
lens = (lens-np.amin(lens))/(np.amax(lens)-np.amin(lens))
lens = exposure.equalize_hist(lens)

fig = plt.figure()
fig.canvas.set_window_title('lens knn')
plt.imshow(np.reshape(lens, imshape_red)[imshape_red[0]//2], cmap = 'hsv')
plt.show()

#%% cluster test
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
communities = asyn_fluidc(G,9, max_iter = 5000)

node_groups = []
for com in communities:  node_groups.append(sorted(com))

color_map = []
class_map = []
for node in G:
    if node in node_groups[0]:
        color_map.append('blue')
        class_map = class_map+[0]
    elif node in node_groups[1]:
        color_map.append('green')
        class_map = class_map+[1]
    elif node in node_groups[2]:
        color_map.append('red')
        class_map = class_map+[2]
    elif node in node_groups[3]:
        color_map.append('yellow')
        class_map = class_map+[3]
    elif node in node_groups[4]:
        color_map.append('orange')  
        class_map = class_map+[4]
    elif node in node_groups[5]:
        color_map.append('pink') 
        class_map = class_map+[5]
    elif node in node_groups[6]:
        color_map.append('brown') 
        class_map = class_map+[6]
    elif node in node_groups[7]:
        color_map.append('white')
        class_map = class_map+[7]
    elif node in node_groups[8]:
        color_map.append('gray')
        class_map = class_map+[8]

fig = plt.figure()
nx.draw(G,  pos, node_color=color_map,node_size = 1)
plt.show()

fig = plt.figure()
fig.canvas.set_window_title('classmap')
plt.imshow(np.reshape(class_map, imshape_red)[imshape_red[0]//2], cmap = 'hsv')
plt.show()

#%%
# creating sub sets (only 1D lenses from now on)

resolution = 40
gain = 7

c_ideal = 5
"""
resolution = 50
gain = 5

t = 0.6
tFix = np.mean(dist)*0.25
w = 0.7
"""
eps = 1E-3 # stretch the borders to get every point

overlap = (gain-1)/gain
my_lens = lens

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
nb = len(subset_borders)
ib = 0
for sb in subset_borders:
    
    
    ib = ib+1
    a=(my_lens>=sb[0])*(my_lens<=sb[1])
    
    if np.sum(a) == 1:
        spc_id = data_red_id[a]
        topo_graph.add_node(i_node, ids = spc_id, height = len(spc_id))
        i_node = i_node+1
    elif np.sum(a) > 1:
        spc_id = data_red_id[a]
        
        ## perform single linkage clustering
        # select correct values from distance matrix
        ids = data_red_id[a]
        idn = len(ids)
        ids = np.meshgrid(ids, ids)
        ida, idb = ids[0].flatten().astype(int), ids[1].flatten().astype(int)
        idd = squareform(np.reshape(dist[ida,idb], (idn,idn)))
        
        Z = linkage(idd, 'single')
        
        n_test = 5
        tList = np.arange(n_test)*1/(n_test-1)
        
        cList = np.asarray([fcluster(Z, t = t*max(Z[:,2]), criterion='distance') for t in tList])
        nList = np.asarray([len(np.unique(c)) for c in cList])
        
        if c_ideal in nList:
            print( ib , " / ", nb, ', #it: ', 0)
            pos = np.where(nList==c_ideal)[0][0]
            cluster = cList[pos]
            n_cluster = nList[pos]
        else:
            # if everything is in here:
            sel = nList<c_ideal
            tMax = tList[sel][0]
            cMax = cList[sel][0]
            nMax = nList[sel][0]
            sel = nList>c_ideal
            tMin = tList[sel][-1]
            cMin = cList[sel][-1]
            nMin = nList[sel][-1]
            
            nm = 0
            while True:
                
                tList = np.arange(n_test)*1/(n_test-1)
                
                d = (tMax-tMin)-2*(tMax-tMin)/(n_test-2)
                tList = list(np.arange(n_test-2)*1/(n_test-3) * d + d + tMin)
                
                cList = np.asarray([cMin] + [fcluster(Z, t = t*max(Z[:,2]), criterion='distance') for t in tList] + [cMax])
                nList = np.asarray([len(np.unique(c)) for c in cList])
                
                if c_ideal in nList:
                    print( ib , " / ", nb, ', #it: ', nm)
                    pos = np.where(nList==c_ideal)[0][0]
                    cluster = cList[pos]
                    n_cluster = nList[pos]
                    break
                elif nm < 20:
                    nm = nm + 1
                    tList = np.asarray([tMin] + tList + [tMax])
                    sel = nList<c_ideal
                    tMax = tList[sel][0]
                    cMax = cList[sel][0]
                    nMax = nList[sel][0]
                    sel = nList>c_ideal
                    tMin = tList[sel][-1]
                    cMin = cList[sel][-1]
                    nMin = nList[sel][-1]
                else:
                    print( ib , " / ", nb, ', #it: ', nm)
                    tList = np.asarray([tMin] + tList + [tMax])
                    pos = np.argmin(np.abs(nList-c_ideal))
                    cluster = cList[pos]
                    n_cluster = nList[pos]
                    break
        for i_c in range(n_cluster):
            id_list = spc_id[cluster==i_c+1]
            if len(id_list) > 0: #more than 1 spc per cluster?
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
            #topo_graph.add_edge(node_a, node_b, weight=len(d)) 
            topo_graph.add_edge(node_a, node_b) 
            delete = 0
    if delete == 1:
        rm = rm+[node_a]
        
n_rm = len(rm)
rm = np.flip(rm)
for r in rm:
    topo_graph.remove_node(r)
print('n_nodes: ', len(topo_graph), " = ", n_nodes , ' - ', n_rm, ' (', n_nodes-n_rm, ') ')

n_nodes = n_nodes-n_rm
pos_topo = nx.spring_layout(topo_graph, weight = None)
fig = plt.figure()
title = 'TOPO - unweighted'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo, node_size = 1)
plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(nx.get_node_attributes(topo_graph, 'height').values()))
plt.title(title)
plt.show()
"""
pos_topo = nx.spring_layout(topo_graph, weight='weight')
fig = plt.figure()
title = 'TOPO - weighted'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo, node_size = 1)
plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(nx.get_node_attributes(topo_graph, 'height').values()))
plt.title(title)
plt.show()
"""
#%%
import community as community_louvain #pip install python-louvain
import matplotlib.cm as cm

dend = community_louvain.generate_dendrogram(topo_graph, weight = 'height')
for level in range(len(dend)):
    partition = community_louvain.partition_at_level(dend, level)
    cmap = cm.get_cmap('hsv', max(partition.values()) + 1)
    plt.figure()
    nx.draw_networkx_nodes(topo_graph, pos_topo, partition.keys(), node_size=2,
                           cmap=cmap, node_color=list(partition.values()))
    
    nx.draw_networkx_edges(topo_graph, pos_topo, alpha=0.1)
    plt.show()
#%%
import itertools

cl = set(partition.values())
part = np.asarray(list(partition.values()))
ids = nx.get_node_attributes(topo_graph, 'ids')
id_topo = np.asarray(list(ids.keys()))
clmap = []
for ic in cl:
    nodes_c = id_topo[part==ic]
    my_nodes = []
    for node in nodes_c: my_nodes.append(list(ids[node]))
    my_nodes = list(set(itertools.chain(*my_nodes)))
    as_map = np.zeros(np.prod(imshape_red))
    as_map[my_nodes] = 1
    
    as_map = as_map * np.sum(data_red, axis = -1)
    as_map = (as_map-np.amin(as_map))/(np.amax(as_map)-np.amin(as_map))
    as_map = exposure.equalize_hist(as_map)
    
    clmap.append(np.reshape(as_map, imshape_red))
    
    plt.figure()
    plt.imshow(np.reshape(as_map, imshape_red)[imshape_red[0]//2])
    plt.show()


#%%

from sklearn.cross_decomposition import PLSRegression

Y = lens_red
X = data_red
data_train = np.reshape(data, (np.prod(imshape), veclen))
pls2 = PLSRegression(n_components=5)
pls2.fit(X, Y)
PLSRegression()
Y_pred = pls2.predict(data_train)

plt.figure()
plt.scatter(*Y_pred.T )
plt.show()

plt.figure()
plt.scatter(*Y.T )
plt.show()


lens = Y_pred[:,0]*Y_pred[:,1]
lens = (lens-np.amin(lens))/(np.amax(lens)-np.amin(lens))
lens = exposure.equalize_hist(lens)

#%%
data = np.reshape(data,(np.prod(imshape), veclen))
data_id = np.arange(np.prod(imshape))

dist = squareform(pdist(data, metric_name))  


#%%
resolution = 100
gain = 15

t = 0.42
tFix = np.mean(dist)*0.25
w = 0.7

eps = 1E-3 # stretch the borders to get every point

overlap = (gain-1)/gain
my_lens = lens

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
        spc_id = data_id[a]
        topo_graph.add_node(i_node, ids = spc_id, height = len(spc_id))
        i_node = i_node+1
    elif np.sum(a) > 1:
        spc_id = data_id[a]
        
        ## perform single linkage clustering
        # select correct values from distance matrix
        ids = data_id[a]
        idn = len(ids)
        ids = np.meshgrid(ids, ids)
        ida, idb = ids[0].flatten().astype(int), ids[1].flatten().astype(int)
        idd = squareform(np.reshape(dist[ida,idb], (idn,idn)))
        
        Z = linkage(idd, 'single')
        #cluster = fcluster(Z, t = (w*t*max(Z[:,2])+(1-w)*tFix), criterion='distance')
        cluster = fcluster(Z, t = 5, criterion='maxclust')
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
            #topo_graph.add_edge(node_a, node_b, weight=len(d)) 
            topo_graph.add_edge(node_a, node_b) 
            delete = 0
    if delete == 1:
        rm = rm+[node_a]
        
n_rm = len(rm)
rm = np.flip(rm)
for r in rm:
    topo_graph.remove_node(r)
print('n_nodes: ', len(topo_graph), " = ", n_nodes , ' - ', n_rm, ' (', n_nodes-n_rm, ') ')

n_nodes = n_nodes-n_rm
pos_topo = nx.spring_layout(topo_graph, weight = None)
fig = plt.figure()
title = 'TOPO - unweighted'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo, node_size = 1)
plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(nx.get_node_attributes(topo_graph, 'height').values()))
plt.title(title)
plt.show()
"""
pos_topo = nx.spring_layout(topo_graph, weight='weight')
fig = plt.figure()
title = 'TOPO - weighted'
fig.canvas.set_window_title(title)
nx.draw(topo_graph, pos_topo, node_size = 1)
plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(nx.get_node_attributes(topo_graph, 'height').values()))
plt.title(title)
plt.show()
"""

#%%
import community as community_louvain #pip install python-louvain
import matplotlib.cm as cm
partition = community_louvain.best_partition(topo_graph, weight = 'height')
print(max(partition.values()) + 1)
cmap = cm.get_cmap('hsv', max(partition.values()) + 1)

nx.draw_networkx_nodes(topo_graph, pos_topo, partition.keys(), node_size=2,
                       cmap=cmap, node_color=list(partition.values()))

nx.draw_networkx_edges(topo_graph, pos_topo, alpha=0.5)
plt.show()

import itertools

cl = set(partition.values())
part = np.asarray(list(partition.values()))
ids = nx.get_node_attributes(topo_graph, 'ids')
id_topo = np.asarray(list(ids.keys()))
clmap = []
for ic in cl:
    nodes_c = id_topo[part==ic]
    my_nodes = []
    for node in nodes_c: my_nodes.append(list(ids[node]))
    my_nodes = list(set(itertools.chain(*my_nodes)))
    as_map = np.zeros(np.prod(imshape_red))
    as_map[my_nodes] = 1
    
    as_map = as_map * np.sum(data_red, axis = -1)
    as_map = (as_map-np.amin(as_map))/(np.amax(as_map)-np.amin(as_map))
    as_map = exposure.equalize_hist(as_map)
    
    clmap.append(np.reshape(as_map, imshape_red))
    
    plt.figure()
    plt.imshow(np.reshape(as_map, imshape_red))
    plt.show()

plt.figure()
plt.imshow(np.reshape(np.sum(data_red, axis = -1), imshape_red))
plt.show()