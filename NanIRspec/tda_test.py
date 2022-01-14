
#%% imports & parameters
import os.path as path
import numpy as np

from IRAFM import IRAFM as ir

"""#####################################################"""
path_project = path.dirname(path.realpath(__file__))
path_final = path.join(path_project, r'resources\2107_weakDataTypeJuly')
headerfile = 'BacVan30_0013.txt'

"""#####################################################"""

my_data = ir(path_final, headerfile)
pos =  [my_file['Caption']=='hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = np.reshape(hyPIRFwd['data'], (1,hyPIRFwd['data'].shape[0],hyPIRFwd['data'].shape[1], hyPIRFwd['data'].shape[2]))


#%%


l2 = np.sqrt(np.sum(data**2, axis = 1))
n_data = data.shape[0]
n_chan = data.shape[1]

dim = np.sqrt(n_data).astype(int)
dim = (dim, dim)

data_norm = np.array([d/n if n > 0 else d*0 for d,n in zip(data, l2)])
data_id = np.arange(n_data)


#%%

from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

n_neighbors = 10
nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(data_norm)
distances, indices = nbrs.kneighbors(data_norm)
weights= 1/(distances+np.mean(distances)*(1E-5))

G = nx.Graph()
G.add_nodes_from(data_id)
[[G.add_edge(node, ind, length = edge_len) for ind, edge_len in zip(node_ind, node_edge)] for node, node_ind, node_edge in zip(data_id,indices,weights)]

pos = nx.spring_layout(G, weight='length')

plt.figure()
nx.draw(G, pos, node_size = 1)
plt.show()

#%%
nSubsets = 40

lensSubset = np.zeros(n_data)-1
pValArray = np.array(list(pos.values()))
pPosArray = np.array(list(pos.keys()))
minX, maxX = np.amin(pValArray[:,0]), np.amax(pValArray[:,0])
minY, maxY = np.amin(pValArray[:,1]), np.amax(pValArray[:,1])

vX = np.std(pValArray[:,0])
vY = np.std(pValArray[:,1])
dX = (maxX - minX)
dY = (maxY - minY)

xy = (vX/dX)/(vY/dY)

ny = np.sqrt(nSubsets/xy)
nx = np.ceil(ny*xy).astype(int)
ny = np.ceil(ny).astype(int)

dX = dX/nx
dY = dY/ny

xList = [[minX+nEdge*dX, minX+(nEdge+1)*dX] for nEdge in range(nx)]
yList = [[minY+nEdge*dY, minY+(nEdge+1)*dY] for nEdge in range(ny)]

for kx in range(nx):
    for ky in range(ny): 
        sel = (pValArray[:,0] >= xList[kx][0]) * (pValArray[:,0] <= xList[kx][1]) * (pValArray[:,1] >= yList[ky][0]) * (pValArray[:,1] <= yList[ky][1])
        lensSubset[pPosArray[sel]] = kx*ny+ky
        
lens_img = lensSubset.reshape(dim)

cmap = plt.cm.jet 
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = (.5, .5, .5, 1.0)
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

plt.figure()
plt.imshow(lens_img, cmap=cmap)
plt.show()        
#%%

# non overlapping subsets

nSubEdge = 7

lensSubset = np.zeros(n_data)-1

pValArray = np.array(list(pos.values()))
pPosArray = np.array(list(pos.keys()))

minX, maxX = np.amin(pValArray[:,0]), np.amax(pValArray[:,0])
minY, maxY = np.amin(pValArray[:,1]), np.amax(pValArray[:,1])
dX = (maxX - minX)/nSubEdge
dY = (maxY - minY)/nSubEdge

xList = [[minX+nEdge*dX, minX+(nEdge+1)*dX] for nEdge in range(nSubEdge)]
yList = [[minY+nEdge*dY, minY+(nEdge+1)*dY] for nEdge in range(nSubEdge)]

for kx in range(nSubEdge):
    for ky in range(nSubEdge): 
        sel = (pValArray[:,0] >= xList[kx][0]) * (pValArray[:,0] <= xList[kx][1]) * (pValArray[:,1] >= yList[ky][0]) * (pValArray[:,1] <= yList[ky][1])
        lensSubset[pPosArray[sel]] = kx*nSubEdge+ky
        
lens_img = lensSubset.reshape(dim)

cmap = plt.cm.jet 
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = (.5, .5, .5, 1.0)
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

plt.figure()
plt.imshow(lens_img, cmap=cmap)
plt.show()
