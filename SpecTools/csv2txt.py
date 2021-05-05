import pandas as pd
import numpy as np
PATH="PDha_1.csv"
df = pd.read_csv( PATH, sep=".",skiprows=[],delimiter="," ) #delimiter for import files, sep for export files
df.to_numpy() #here you can choose export file like excel latex (numpy for txt)
print("dftxt",df.shape)
np.savetxt("txt file of raw.txt",df, delimiter="\t")
