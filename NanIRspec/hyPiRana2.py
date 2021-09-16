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

from IRAFM import IRAFM as ir

"""#####################################################"""
path_project = path.dirname(path.realpath(__file__))
path_final = path.join(path_project, 'resources', '2108_Control30')
#path_final = '/home/tanoshimi/Programming/python/collaborations/TAEUBER_DANIELA/irafmimaging/NanIRspec/resources/2107_weakDataTypeJuly/'
headerfile = 'Control30_0016.txt'
#headerfile = 'data.npy'

"""#####################################################"""

my_data = ir(path_final, headerfile)
#my_data = np.load(path.join(path_final, headerfile))
pos = [my_file['Caption'] == 'hyPIRFwd' for my_file in my_data['files']]
hyPIRFwd = np.array(my_data['files'])[pos][0]
data = np.reshape(hyPIRFwd['data'], (hyPIRFwd['data'].shape[0] *
                  hyPIRFwd['data'].shape[1], hyPIRFwd['data'].shape[2]))
my_sum = np.sum(data, axis=1)
data_train = data[my_sum != 0]

print("nice")
