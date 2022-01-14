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
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import sklearn

from IRAFM import IRAFM as ir
"""#####################################################"""
path_project = path.dirname(path.realpath(__file__))

#data = np.load(r'C:\Users\ungersebastian\Programmierung\py\raman3D.npy')[:,:,:,200:400]

#################


"""#####################################################"""
path_project = path.dirname(path.realpath(__file__))
path_final = path.join(path_project, r'resources\2107_weakDataTypeJuly')
headerfile = 'BacVan30_0013.txt'

"""#####################################################"""

my_data = ir(path_final, headerfile)
pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = np.reshape(hyPIRFwd['data'], (1,hyPIRFwd['data'].shape[0],hyPIRFwd['data'].shape[1], hyPIRFwd['data'].shape[2]))
data = data + 1E-10


################


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

    data_red = fun(data_red.reshape(*reshape), axis = tuple(np.arange(1,1+len(ww)*2,2).astype(int)))
    
    return data_red
"""
# for trying, rm later

ww = [2,4,4,2]
thresh = 0.005

pool_mean = poolNd(data, ww, np.mean)
pool_median = poolNd(data, ww, np.median)

d = np.abs((pool_mean-pool_median)/(pool_mean+pool_median+np.mean(pool_mean)*1E-6))

pool_mean[d>=thresh]=pool_median[d>=thresh]

data = pool_mean

nZ,*imshape, vChan = data.shape

fig = plt.figure()
fig.canvas.set_window_title('original data')
plt.imshow(np.sum(data[nZ//2], axis = -1))
plt.show()
"""
# creating the lens

# downscaling and removal of some suspicios datapoints (ie spikes)

metric_name = 'cosine' # euclidean, seuclidean, correlation, cosine
nLensSteps = 10

ww = [1,1,1,1]
thresh = 0.005

pool_mean = poolNd(data, ww, np.mean)
pool_median = poolNd(data, ww, np.median)

d = np.abs((pool_mean-pool_median)/(pool_mean+pool_median+np.mean(pool_mean)*1E-6))
print('amount of flattend positions: ',np.sum(d>=thresh)/len(d.flatten()))
pool_mean[d>=thresh]=pool_median[d>=thresh]

fig = plt.figure()
fig.canvas.set_window_title('scaled down data')
plt.imshow(np.sum(pool_mean[int((nZ/ww[0])//2)], axis = -1))
plt.show()

*imshape_red, vChan_red = pool_mean.shape


data_red = np.reshape(pool_mean, (np.prod(pool_mean.shape[:-1]),pool_mean.shape[-1]))
data_red_id = list(np.arange(len(data_red)))

dist = squareform(pdist(data_red, metric_name))
#np.fill_diagonal(dist, val = np.inf)
lensList = []

mymax = np.amin([nLensSteps, dist.shape[0]])
sList = [np.sum(dist[i][np.argsort(dist[i])[:mymax]]) for i in data_red_id]
iStart = data_red_id[np.argmin(sList)]
lensList.append(data_red_id[iStart])
distList = [0]
distOld = 0
dist_lens = dist.__copy__()
id_lens = list(np.array(data_red_id).__copy__())
data_red_id = np.array(data_red_id)
while(dist_lens.shape[0]>1):
    iOld = iStart
    dList = list(dist_lens[iOld])
    dList = dList[:iOld]+dList[iOld+1:]
    iList = np.argsort(dList)[:mymax]
    
    id_lens.remove(id_lens[iOld])
    dist_lens = np.delete(np.delete(dist_lens, iOld, axis = 0), iOld, axis = 1)
    
    sList = [np.sum(dist_lens[i][np.argsort(dist_lens[i])[:mymax]]) for i in iList]
    
    iStart = iList[np.argmin(sList)]
    lensList.append(id_lens[iStart])
    distOld = distOld + dist[iOld, id_lens[iStart]]
    distList.append(distOld)

lensIm = np.zeros(len(lensList))
lensIm[lensList] = distList


#interpolate data to data_red lens values

iMax = len(distList)
new_lens = []
spc = np.reshape(data, (nZ*np.prod(imshape), vChan))

eps = 1E-10

for i_s, s in enumerate(spc):

    s = np.reshape(s,(1,vChan))
    dist_s = cdist(s, data_red, metric_name)[0]
    
    min_s = np.argmin(dist_s)
    p_min = lensList.index(min_s)
    
    if p_min == 0:
        vec_min = [p_min, p_min+1]
        dist_s = dist_s[np.asarray(lensList)[vec_min]]
        dist_n = dist_s[1]-dist_s[0]
        
        o = dist_n / distList[1]
        if o < 1:
            dist_s = dist_s / (np.sum(dist_s) + eps)
            
            p = distList[vec_min[0]] * dist_s[1] + distList[vec_min[1]] * dist_s[0]
        else:
            dp = distList[1]-distList[0]
            p = distList[0] - (dist_n-dp)
            
    elif p_min == iMax-1:
        vec_min = [p_min-1, p_min]
        
        dist_s = dist_s[np.asarray(lensList)[vec_min]]
        dist_n = dist_s[0]-dist_s[1]
        o = dist_n / distList[-1]
        if o < 1:
            dist_s = dist_s / (np.sum(dist_s) + eps)
            p = distList[vec_min[0]] * dist_s[1] + distList[vec_min[1]] * dist_s[0]
        else:
            dp = distList[-1]-distList[-2]
            p = distList[-1] + np.abs(dist_n-dp)
    
    else:
        vec_min = [p_min-1, p_min, p_min+1]
        dist_s = dist_s[np.asarray(lensList)[vec_min]]
        
        amin = np.argsort(dist_s)[:2]
        
        dist_s = np.asarray(dist_s)[amin]
        dist_s = dist_s / (np.sum(dist_s) + eps)
        
        p = distList[vec_min[amin[0]]] * dist_s[1] + distList[vec_min[amin[1]]] * dist_s[0]
        
    new_lens.append(p)
    
new_lens = np.array(new_lens)
new_lens[np.isnan(new_lens)==True] = 0
new_lens = (new_lens - np.amin(new_lens)) / (np.abs(np.amax(new_lens) - np.amin(new_lens))+eps)     
new_lens[new_lens == np.NAN] = 0
new_lens_im = np.reshape(new_lens, (nZ,*imshape))
fig = plt.figure()
fig.canvas.set_window_title('original data')
plt.imshow(new_lens_im[nZ//2])
plt.show()

#%%
plt.figure()
plt.imshow(np.reshape(lensIm, imshape_red)[int((nZ/ww[0])//2)], cmap = 'hsv')
plt.show()

plt.figure()
plt.imshow(np.reshape(new_lens, (nZ,*imshape))[nZ//2], cmap = 'hsv')
plt.show()
lens = (lensIm - np.amin(lensIm)) / (np.amax(lensIm) - np.amin(lensIm))

#%%
bins = 100

b0 = np.round(bins * np.prod(imshape_red) / np.prod(imshape)*nZ).astype(int)


plt.figure()
plt.hist(new_lens, bins = bins)
plt.show()

plt.figure()
plt.hist(lensIm, bins = b0)
plt.show()

#%% pca lens?

v = 2

ds = np.reshape(data, (np.prod(data.shape[:-1]), data.shape[-1])).__copy__()
s = np.sum(ds, axis = 1)
ds = np.array([i_ds/i_s if i_s > 0 else i_ds*0 for i_ds, i_s in zip(ds, s)])
mean = np.mean(ds, axis = 0)
ds = ds - mean

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
sc = pca.fit_transform(ds)

plt.figure()
plt.imshow(np.reshape(sc, (nZ,*imshape))[nZ//2], cmap = 'hsv')
plt.show()

sv = np.std(sc)
me = np.median(sc)

mask = np.zeros(sc.shape)
mask[sc<me-v*sv]=1
mask[sc>me+v*sv]=1


pca = PCA(n_components=1)
pca.fit(ds[mask.flatten()==0])
sc = pca.transform(ds)

sv = np.std(sc)
me = np.median(sc)

sc[sc<me-v*sv]=me-v*sv
sc[sc>me+v*sv]=me+v*sv

plt.figure()
plt.imshow(np.reshape(sc, (nZ,*imshape))[nZ//2], cmap = 'hsv')
plt.show()

plt.figure()
plt.imshow(np.reshape((sc.flatten()-np.amin(sc)) * new_lens, (nZ,*imshape))[nZ//2], cmap = 'hsv')
plt.show()

plt.figure()
plt.hist(sc, bins = bins)
plt.show()

plt.figure()
plt.hist((sc.flatten()-np.amin(sc)) * new_lens, bins = bins)
plt.show()
#%%
# creating sub sets (only 1D lenses from now on)

resolution = 250
gain = 15
c_ideal = 6

data = np.reshape(data,(np.prod(data.shape[:-1]), data.shape[-1]))
data_id = np.arange(len(data))

eps = 1E-3 # stretch the borders to get every point

overlap = (gain-1)/gain
my_lens = new_lens

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
    break1 = 0
    
    ib = ib+1
    a=(my_lens>=sb[0])*(my_lens<=sb[1])
    
    if np.sum(a) == 1:
        spc_id = data_id[a]
        topo_graph.add_node(i_node, ids = spc_id, height = len(spc_id))
        i_node = i_node+1
    elif np.sum(a) > 1:
        spc_id = data_id[a]
        
        ## perform single linkage clustering
        # select correct values from distance matrix
       
        
        """ if n_spc is small, this is a much faster way!
        ids = data_red_id[a]
        idn = len(ids)
        ids = np.meshgrid(ids, ids)
        ida, idb = ids[0].flatten().astype(int), ids[1].flatten().astype(int)
        idd = squareform(np.reshape(dist[ida,idb], (idn,idn)))
        """
        idd = pdist(data[a], metric_name)
        
        idd[np.isnan(idd)] = 0
        
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
            if np.sum(sel) == 0:
                tMin = tList[-1]
                cMin = cList[-1]
                nMin = nList[-1]
            else:
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
                    if np.sum(sel) == 0:
                        tMin = tList[-1]
                        cMin = cList[-1]
                        nMin = nList[-1]
                    else:
                            
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
    nx.draw_networkx_nodes(topo_graph, pos_topo, partition.keys(), node_size=10,
                           cmap=cmap, node_color=list(partition.values()))
    
    nx.draw_networkx_edges(topo_graph, pos_topo, alpha=0.4)
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
    if len(my_nodes)>1:
        
        as_map = np.zeros(nZ*np.prod(imshape))
        as_map[my_nodes] = 1
        
        as_map = as_map * np.sum(data, axis = -1)
        as_map = (as_map-np.amin(as_map))/(np.amax(as_map)-np.amin(as_map))
        as_map = exposure.equalize_hist(as_map)
        
        clmap.append(np.reshape(as_map, (nZ, *imshape)))
        
        plt.figure()
        plt.imshow(np.reshape(as_map, (nZ, *imshape))[nZ//2])
        plt.show()


#%%
sList = []
for ic in cl:
    nodes_c = id_topo[part==ic]
    my_nodes = []
    for node in nodes_c: my_nodes.append(list(ids[node]))
    my_nodes = list(set(itertools.chain(*my_nodes)))
    if len(my_nodes)>1:
        
        as_map = np.zeros(np.prod((nZ, *imshape)))
        as_map[my_nodes] = 1
        sList.append(np.mean(data[as_map==1], axis = 0))
        plt.figure()
        plt.plot(np.mean(data[as_map==1], axis = 0))
        plt.show()
"""        
sList = np.array(sList)

from sklearn.decomposition import non_negative_factorization

H, W, n =  non_negative_factorization(data, n_components = sList.shape[0], init='custom', update_H=False, H=sList, max_iter = 5000)
"""
#%%
import NanoImagingPack as nip
res = H.__copy__().T
for r in res:
    plt.figure()
    plt.imshow(np.reshape(r, (nZ, *imshape))[nZ//2])
    plt.show()
    
#%%

arr = np.reshape(res, (res.shape[0], *imshape))
nip.imsave(nip.image(arr),r'C:\Users\ungersebastian\Programmierung\py\irafmimaging\im.tif')