

from IRAFM import IRAFM as ir
from os.path import join
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

path_file = join(r'C:')
header_file1 = 'BacVan30_0011.txt'
header_file2 = 'BacVan60_0013.txt'



Data1 = ir(path_file, header_file1)
spc1 = Data1.return_spc()
exten1 = Data1.extent()
#print(spc1)


Data2 = ir(path_file, header_file2)
spc2 = Data2.return_spc()
exten2 = Data2.extent()
#print('second')
#print(spc2.shape)


figure = plt.figure()
ax = plt.subplot(111)
plt.cmap = 'ocean'

#Alpha for the transperancy
image1 = plt.colorbar(ax.imshow(spc1, extent = exten1, interpolation = 'bicubic', cmap='viridis'))
image2 = plt.colorbar(ax.imshow(spc2, extent = exten2, interpolation = 'bicubic', cmap='viridis', alpha = 0.7))



#[x, y] = image1
#R = image1(x[6], y[20])
#G = image1(x[60], y[42])
#B = image1(x[12], y[3])