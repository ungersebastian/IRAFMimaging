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
import functools
from scipy.signal import medfilt

from IRAFM import IRAFM as ir
"""#####################################################"""
path_project = path.dirname(path.realpath(__file__))

#data = np.load(r'C:\Users\ungersebastian\Programmierung\py\raman3D.npy')[:,:,:,200:400]

#################


"""#####################################################"""
path_project = path.dirname(path.realpath(__file__))
#path_final = path.join(path_project, r'resources\2107_weakDataTypeJuly')
#headerfile = 'BacVan30_0013.txt'

"""#####################################################"""

#my_data = ir(path_final, headerfile)
#pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
#hyPIRFwd = np.array(my_data['files'])[pos][0]
#data = np.reshape(hyPIRFwd['data'], (1,hyPIRFwd['data'].shape[0],hyPIRFwd['data'].shape[1], hyPIRFwd['data'].shape[2]))
#data = data + 1E-10
#data = np.load(r'N:\Daten_Promotions_Sebastian\raman3D.npy')[:,60:-60,60:-60,260:340]
data = np.load(r'N:\Daten_Promotions_Sebastian\raman3D.npy')[:,:,:,180:420]
if data.ndim < 4:
    data = np.reshape(data, (*list(np.ones(4-data.ndim).astype(int)), *data.shape))

#%%

    
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


#%%

def fun_dist_lens(data, metric_name='cosine', nLensSteps=10):
    data_rs = np.reshape(data, (np.prod(data.shape[:-1]),data.shape[-1]))
    data_id = list(np.arange(len(data_rs)))
    
    dist = squareform(pdist(data_rs, metric_name))
    
    lensList = []
    
    mymax = np.amin([nLensSteps, dist.shape[0]])
    sList = [np.sum(dist[i][np.argsort(dist[i])[:mymax]]) for i in data_id]
    iStart = data_id[np.argmin(sList)]
    lensList.append(data_id[iStart])
    distList = [0]
    distOld = 0
    dist_lens = dist.__copy__()
    id_lens = list(np.array(data_id).__copy__())
    data_id = np.array(data_id)
    
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
    #lensIm[lensList] = np.array(distList)[lensList]
    lensIm[lensList] = distList
    return distList, lensIm, lensList

#%%
# creating the lens

# downscaling and removal of some suspicios datapoints (ie spikes)

metric_name = 'cosine' # euclidean, seuclidean, correlation, cosine
nLensSteps = 10
thresh = 0.010
ww = [2,3,3,1]

if functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, list(map(int,ww)), np.ones(data.ndim).astype(int)), True) : 
    distList, lensIm, lensList = fun_dist_lens(data, metric_name, nLensSteps)
    
    plt.figure()
    plt.imshow(np.reshape(lensIm, (nZ,*imshape))[int((nZ/ww[0])//2)], cmap = 'hsv')
    plt.show()
else :
    pool_mean = poolNd(data, ww, np.mean)
    pool_median = poolNd(data, ww, np.median)
    
    d = np.abs((pool_mean-pool_median)/(pool_mean+pool_median+np.mean(pool_mean)*1E-6))
    print('amount of flattend positions: ',np.sum(d>=thresh)/len(d.flatten()))
    pool_mean[d>=thresh]=pool_median[d>=thresh]
    
    *imshape_red, vChan_red = pool_mean.shape

    data_red = np.reshape(pool_mean, (np.prod(pool_mean.shape[:-1]),pool_mean.shape[-1]))
    
    distList, lensIm, lensList = fun_dist_lens(data_red, metric_name, nLensSteps)
    
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
    
    plt.figure()
    plt.imshow(np.reshape(new_lens, (nZ,*imshape))[int(nZ//2)], cmap = 'hsv')
    plt.show()
    
"""    
#%% start here
distList, lensIm, lensList, new_lens
imshape_red, nZ, imshape

#%%
"""
def get_coords(imshape, index):
    
    n_dim = len(imshape)
    t = index
    xl = []
    
    for it in range(n_dim):
        p = np.prod(imshape[it+1:])
        x = int(t // p)
        t = t - x * p
        xl.append(x)
    
    return tuple(xl)
def extract(im, coords, widths):
    shape = im.shape
    tup = (slice(max((0,cc-ww)), min((cc+ww+1, ss)), 1) for cc, ww, ss in zip(coords, widths, shape))
    return list(np.asarray(im[tuple(tup)]).flatten())
def flatten_unique(t):
    return list(np.sort(np.unique([item for sublist in t for item in sublist])))

dim = (nZ, *imshape)
id_im = np.reshape(np.arange(np.prod(dim)),dim)


spc = np.reshape(data.__copy__(), (nZ*np.prod(imshape), vChan))
extra_width = (0,1,1)
step = 3
window = 1000

pre_sort = np.argsort(new_lens)

id_list = []
d_list = [0]
ds_list = [0]

d_old = 0

while len(pre_sort)>0:

    # das erste Element ist das in der ersten Zeile

    i_old = 0
    
    id_sort = pre_sort[:window+1]
    
    #### ND hinzufügen

    # um die vorgewählten Spektren werden die räumlichen Nachbarn hinzugefügt
    extra_id = [get_coords((nZ,*imshape), ids) for ids in id_sort]
    list_extra_id = [extract(id_im, tt, extra_width) for tt in extra_id]
    list_extra_id = flatten_unique(list_extra_id)
    # es werden nur die Spektren genommen, welche noch verfügbar sind
    list_extra_id = list(set(pre_sort).intersection(list_extra_id))
    
    id_sort = np.array(list_extra_id)

    #### Ende ND
    
    sort_dist = squareform(pdist(spc[id_sort], metric = metric_name))
    np.fill_diagonal(sort_dist, np.infty)

    for i_step in range(step):
        
        # alte ID hinzufügen
        t = id_sort[i_old]
        id_list.append(t)
        
        # 1 aktuelle Achse nehmen und auf neue Größe reduzieren
        
        ax = list(sort_dist[:,i_old])
        ax = ax[:i_old]+ax[i_old+1:]
        if len(ax) > 0:
            
            # Position neues Minimum auf redzierter Achse
            i_new = np.argmin(ax)
            d_new = ax[i_new]
            
            
            # alte Einträge löschen
            sort_dist = np.delete(np.delete(sort_dist, i_old, axis = 0), i_old, axis = 1)
            id_sort = np.delete(id_sort, np.where(id_sort == t)[0][0])
            pre_sort = np.delete(pre_sort, np.where(pre_sort == t)[0][0])
            
            # Distanzen hinzufügen
            d_list.append(d_new)
            ds_list.append(d_old+d_new)
            
            # aktualiserung der Einträge
            d_old = d_old + d_new
            i_old = i_new
        else:
            # alte Einträge löschen
            pre_sort = np.delete(pre_sort, i_old)
            break
    if len(pre_sort)>0:
        # an der Stelle wurden step-1 Elemente hinzugefügt     
        # das nächste Element ist Startpunkt der nächsten step Elemente
        pre_sort = list(pre_sort)
        index = pre_sort.index(id_sort[i_old])
        pre_sort = np.array([pre_sort[index]] + pre_sort[:index] + pre_sort[index+1:])
    
    print(float(int(1000*len(id_list)/len(spc)))/10)
    
#
#%%

spc = np.reshape(data.__copy__(), (nZ*np.prod(imshape), vChan))
extra_width = (0,5,5)
step = 1
window = 50

pre_sort = np.argsort(new_lens)

id_list = []
d_list = [0]
ds_list = [0]

d_old = 0

while len(pre_sort)>0:

    # das erste Element ist das in der ersten Zeile

    i_old = 0
    
    id_sort = pre_sort[:window+1]
    
    #### ND hinzufügen

    # um die vorgewählten Spektren werden die räumlichen Nachbarn hinzugefügt
    extra_id = [get_coords((nZ,*imshape), ids) for ids in id_sort]
    list_extra_id = [extract(id_im, tt, extra_width) for tt in extra_id]
    list_extra_id = flatten_unique(list_extra_id)
    # es werden nur die Spektren genommen, welche noch verfügbar sind
    list_extra_id = list(set(pre_sort).intersection(list_extra_id))
    
    id_sort = np.array(list_extra_id)

    #### Ende ND
    
    sort_dist = squareform(pdist(spc[id_sort], metric = metric_name))
    np.fill_diagonal(sort_dist, np.infty)

    for i_step in range(step):
        
        # alte ID hinzufügen
        t = id_sort[i_old]
        id_list.append(t)
        
        # 1 aktuelle Achse nehmen und auf neue Größe reduzieren
        
        ax = list(sort_dist[:,i_old])
        ax = ax[:i_old]+ax[i_old+1:]
        if len(ax) > 0:
            
            # Position neues Minimum auf redzierter Achse
            i_new = np.argmin(ax)
            d_new = ax[i_new]
            
            
            # alte Einträge löschen
            sort_dist = np.delete(np.delete(sort_dist, i_old, axis = 0), i_old, axis = 1)
            id_sort = np.delete(id_sort, np.where(id_sort == t)[0][0])
            pre_sort = np.delete(pre_sort, np.where(pre_sort == t)[0][0])
            
            # Distanzen hinzufügen
            d_list.append(d_new)
            ds_list.append(d_old+d_new)
            
            # aktualiserung der Einträge
            d_old = d_old + d_new
            i_old = i_new
        else:
            # alte Einträge löschen
            pre_sort = np.delete(pre_sort, i_old)
            break
    if len(pre_sort)>0:
        # an der Stelle wurden step-1 Elemente hinzugefügt     
        # das nächste Element ist Startpunkt der nächsten step Elemente
        pre_sort = list(pre_sort)
        index = pre_sort.index(id_sort[i_old])
        pre_sort = np.array([pre_sort[index]] + pre_sort[:index] + pre_sort[index+1:])
    
    print(float(int(1000*len(id_list)/len(spc)))/10)
#%%
print(len(id_list))
print(len(np.unique(id_list)))


#%%


#%%
bins = 100

b0 = np.round(bins * np.prod(imshape_red) / np.prod(imshape)*nZ).astype(int)

plt.figure()
plt.hist(new_lens, bins = bins)
plt.show()

plt.figure()
plt.hist(lensIm, bins = b0)
plt.show()

plt.figure()
plt.hist(ds_list, bins = b0)
plt.show()

#%%

new_im = np.zeros(len(ds_list))
new_im[id_list] = ds_list
new_im = np.reshape(new_im, (nZ, *imshape))

fig = plt.figure()
fig.canvas.set_window_title('intensity image')
plt.imshow(np.sum(data[nZ//2], axis = -1) , cmap = 'gray')
plt.show()

fig = plt.figure()
fig.canvas.set_window_title('lens, pooling')
plt.imshow(np.reshape(lensIm, imshape_red)[int((nZ/ww[0])//2)], cmap = 'gray')
plt.show()

fig = plt.figure()
fig.canvas.set_window_title('lens, interpolation')
plt.imshow(np.reshape(new_lens, (nZ,*imshape))[nZ//2], cmap = 'gray')
plt.show()

fig = plt.figure()
fig.canvas.set_window_title('lens, sorting')
plt.imshow(new_im[nZ//2], cmap = 'gray')
plt.show()

fig = plt.figure()
fig.canvas.set_window_title('lens, pooling')
plt.imshow(np.reshape(lensIm, imshape_red)[int((nZ/ww[0])//2)], cmap = 'hsv')
plt.show()

fig = plt.figure()
fig.canvas.set_window_title('lens, interpolation')
plt.imshow(np.reshape(new_lens, (nZ,*imshape))[nZ//2], cmap = 'hsv')
plt.show()

fig = plt.figure()
fig.canvas.set_window_title('lens, sorting')
plt.imshow(new_im[nZ//2], cmap = 'hsv')
plt.show()

#%%

np.save("pooling_2-3-3-1_0-1-1_3_1000.npy", lensIm)
np.save("interpolation_lens_2-2-2-1_0-1-1_3_1_3_1000.npy", new_lens)
np.save("sorting_lens_2-3-3-1_0-1-1_3_1_3_1000.npy", new_im)

#%%

metric_name = 'cosine'
new_im = np.load("sorting_lens_2-3-3-1_0-1-1_3_1_3_1000.npy")


fig = plt.figure()
fig.canvas.set_window_title('lens, sorting')
plt.imshow(new_im[nZ//2], cmap = 'hsv')
plt.show()

plt.figure()
plt.hist(new_im.flatten(), bins = 1000)
plt.show()

from scipy.ndimage import median_filter
from scipy.ndimage.filters import convolve

sorting = np.argsort(new_im.flatten())
dl = np.sort(new_im.flatten())
dl = dl[1:] - dl[:-1]

fSize = 5
thresh = 0.1
q1 = 1
q2 = 2
eps = 1E-10

weight = np.full((fSize), 1.0/fSize)
dl_med = median_filter(dl, fSize)
dl_mea = convolve(dl, weight)
t = np.abs((dl_mea-dl_med)/(dl_med+dl_mea+eps))
dl_mea[t>thresh] = dl_med[t>thresh]
mea = np.mean(dl_mea)
std = np.std(dl_mea)
sel = dl_mea >= mea+q1*std
off = dl_mea[sel]
mi, ma = min(off), max(off)
off = ((off-mi)/(ma-mi)) * ((q2-q1)*std) + mea+q1*std
dl_mea[sel]=off

old = 0
lens = [0]
for d in dl_mea:
    old = old+d
    lens.append(old)
#%%    
plt.figure()
plt.hist(lens, bins = 100)
new_im_flat = np.zeros(len(lens))
new_im_flat[sorting] = lens
new_im_flat = np.reshape(new_im_flat, (nZ, *imshape))

fig = plt.figure()
fig.canvas.set_window_title('lens, sorting')
plt.imshow(new_im_flat[nZ//2], cmap = 'hsv')
plt.show()

#%%
resolution_windows = 4
resolution = 60*resolution_windows

gain = 6
method = 'ward'
t = 10
limit = 5


my_lens = new_im_flat.flatten()

resolution = np.ceil(resolution/resolution_windows).astype(int)*resolution_windows

eps = 1E-3 # stretch the borders to get every point
overlap = (gain-1)/gain

data = np.reshape(data,(np.prod(data.shape[:-1]), data.shape[-1]))
data_id = np.arange(len(data))

minmax = np.amax(my_lens, axis = 0) - np.amin(my_lens, axis = 0)
eps = (minmax/resolution)*eps
mins = np.amin(my_lens, axis = 0)

c_width = (minmax+2*eps)/(1+(resolution-1)*(1-overlap))
c_dist  = c_width*(1-overlap)

subset_borders = [[mins - eps + r*c_dist, mins - eps + r*c_dist + c_width] for r in range(resolution)]
border_select = [[i*resolution//resolution_windows+j for i in range(resolution_windows)] for j in range(resolution//resolution_windows)]

topo_graph = nx.Graph()
i_node = 0
ib = 0


for bs in border_select:
    ib = ib+1
    
    my_border = [ (my_lens>=subset_borders[i_sb][0])*(my_lens<=subset_borders[i_sb][1]) for i_sb in bs]
    my_border = np.sum(np.array(my_border), axis = 0)>0
    
    a = my_border
    
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
        Z = linkage(idd, method)
        
        #cluster analyse
        lens_dist = np.median(my_lens[a][1:]-my_lens[a][:-1])
        c = fcluster(Z, t = t*lens_dist, criterion='distance')
        n0=len(np.unique(c))
        
        # nur cluster nehmen, welche groß geng sind
        c_uni = np.unique(c)
        c_o = [np.sum(c==ic)<limit for ic in c_uni]
        c_n = [np.sum(c==ic)>=limit for ic in c_uni]
        c_out = c_uni[c_o]
        c_uni = c_uni[c_n]
        if len(c_uni)>0:
            for co in c_out:
                c[c==co] = 0
            
            # cluster mediane bilden und spektren spektral median filtern
            
            spc_filt = medfilt(data[a].__copy__(), (1,3))
            c_spc = np.array([np.median(spc_filt[c==cu], axis = 0) for cu in c_uni])
            
            co_id = np.arange(len(c))[c==0]
            co_spc = spc_filt[c==0]
            
            # übrige spektren einem der großen cluster zuordnen
            
            co_c_dist = cdist(co_spc, c_spc, metric = metric_name)
            mins = np.argmin(co_c_dist, axis = 1)
            c[co_id] = c_uni[mins]
            
            print(n0, " -> ", len(c_uni), ": ",[np.sum(c==cu) for cu in c_uni])
            
            """
            c_spc = [np.median(spc_filt[c==cu], axis = 0) for cu in c_uni]
            c_std = [np.std(spc_filt[c==cu], axis = 0) for cu in c_uni]
            plt.figure()
            for cs, ct in zip(c_spc, c_std):
                plt.plot(cs)
                #plt.fill_between(np.arange(len(cs)), cs+ct, cs-ct, alpha = 0.2)
            plt.show()
            """
            
        else:
            print(n0, ", no unique clusters")
        for i_c in np.unique(c):
            id_list = spc_id[c==i_c]
            topo_graph.add_node(i_node, ids = id_list, height = len(id_list))
            i_node = i_node+1
    print( ib , " / ", resolution//resolution_windows)

      
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
            #topo_graph.add_edge(node_a, node_b) 
            delete = 0
    if delete == 1:
        rm = rm+[node_a]
        
n_rm = len(rm)
rm = np.flip(rm)
for r in rm:
    topo_graph.remove_node(r)
print('n_nodes: ', len(topo_graph), " = ", n_nodes , ' - ', n_rm, ' (', n_nodes-n_rm, ') ')

n_nodes = n_nodes-n_rm
#%%
for i in range(20):
    
    pos_topo = nx.spring_layout(topo_graph, weight = None)
    fig = plt.figure()
    title = 'TOPO - unweighted'
    fig.canvas.set_window_title(title)
    #nx.draw(topo_graph, pos_topo, node_size = 1)
    nx.draw_networkx_nodes(topo_graph, pos_topo, node_size=1)
    nx.draw_networkx_edges(topo_graph, pos_topo, alpha=0.4)
    
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

fig = plt.figure()
title = 'TOPO - unweighted'
fig.canvas.set_window_title(title)
nx.draw_networkx_nodes(topo_graph, pos_topo, node_size=1)
nx.draw_networkx_edges(topo_graph, pos_topo, alpha=0.4)

plt.scatter(*np.transpose(np.array(list(pos_topo.values()))), c = list(nx.get_node_attributes(topo_graph, 'height').values()))
plt.title(title)
plt.show()

#%%
import community as community_louvain #pip install python-louvain
import matplotlib.cm as cm

partition = community_louvain.best_partition(topo_graph, weight = 'height')
plt.figure()
nx.draw_networkx_nodes(topo_graph, pos_topo, partition.keys(), node_size=10,
                       cmap = cm.get_cmap('hsv', max(partition.values()) + 1), node_color=list(partition.values()))

nx.draw_networkx_edges(topo_graph, pos_topo, alpha=0.4)
plt.show()
"""
dend = community_louvain.generate_dendrogram(topo_graph, weight = 'height')
for level in range(len(dend)):
    partition = community_louvain.partition_at_level(dend, level)
    cmap = cm.get_cmap('hsv', max(partition.values()) + 1)
    plt.figure()
    nx.draw_networkx_nodes(topo_graph, pos_topo, partition.keys(), node_size=10,
                           cmap=cmap, node_color=list(partition.values()))
    
    nx.draw_networkx_edges(topo_graph, pos_topo, alpha=0.4)
    plt.show()
   """
    
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
        #plt.figure()
        #plt.plot(np.mean(data[as_map==1], axis = 0))
        #plt.show()
      
sList = np.array(sList)

from sklearn.decomposition import non_negative_factorization

H, W, n =  non_negative_factorization(data, n_components = sList.shape[0], init='custom', update_H=False, H=sList, max_iter = 5000)


import NanoImagingPack as nip
res = H.__copy__().T
for r in res:
    plt.figure()
    plt.imshow(np.reshape(r, (nZ, *imshape))[nZ//2])
    plt.show()

#%%

    
#%%

arr = np.reshape(res, (res.shape[0], nZ, *imshape))
nip.imsave(nip.image(arr[:, nZ//2]),r'M:\Downloads\im.tif')

#%%

inten = np.sum(data, axis = -1)
resInt = np.reshape([r*inten for r in res], (res.shape[0], nZ, *imshape))
nip.imsave(nip.image(resInt[:, nZ//2]),r'M:\Downloads\imres.tif')