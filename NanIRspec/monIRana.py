"""" monIRana is a program for analysis of a set of spectral data provided in a tabulated textfile
Created on 4 july 2021
@author: Mohammad Soltaninezhad
Supervisor: Daniela Täuber

monIRana can do:
- calculate mean spectra of the complete data set
- run a PCA on the data set and provide plots of results
"""


from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
path =r"F:/daniela/retina/NanIRspec/resources/BacVan30Control30_OPTIR.txt"
df = pd.read_csv(path,skiprows=[0],delimiter="\t" )
df.to_numpy()
class PCA_calculator:
    def __init__(self, path):
        self.path = path
        self.read_data()
    def read_data(self): #read data
        df = pd.read_csv( self.path,skiprows=[0],delimiter="\t" )  # in skiprows write the rows which contain text in order to eliminate them
        df.to_numpy()
        data = df.values.T
        my_spc=data[1:]  #spectra
        my_wl = data[0]  # wavelength
        self.my_wl = np.nan_to_num(my_wl)
        self.my_spc = np.nan_to_num(my_spc)
        return self.my_spc , self.my_wl
    def pca(self,ncomp):
        ncomp=int(ncomp)
        my_sum = np.sum( self.my_spc , axis=1 ) #sum of every axis
        my_sum = my_sum[my_sum != 0]
        spc_norm = np.array( [spc / s for spc, s in zip( my_spc, my_sum )] ) #normalization of spectra
        mean_spc = np.mean( spc_norm, axis=0 )
        std_spc = np.std( spc_norm, axis=0 )
        model = PCA( ncomp )
        transformed_data = model.fit( spc_norm - mean_spc ).transform( spc_norm - mean_spc ).T
        loadings = model.components_
        return spc_norm , my_sum ,mean_spc,std_spc,transformed_data,loadings
    def pca_var(self,ncomp): #draw variance percentage
        ncomp = int( ncomp )
        pca_var = PCA( ncomp )
        data_var = pca_var.fit( spc_norm - mean_spc ).transform( spc_norm - mean_spc ).T
        pca_var.fit( data_var )
        pca_data_var = pca_var.transform( data_var )
        per_var = np.round( pca_var.explained_variance_ratio_ * 100, decimals=1 )
        labels = ['pc' + str( x ) for x in range( 1, len( per_var ) + 1 )]
        plt.bar( x=range( 1, len( per_var ) + 1 ), height=per_var, tick_label=labels )
        plt.ylabel( "percantage of explained variance" )
        plt.xlabel( "Principle Components" )
        plt.title( "Bar plot" )
        plt.show()
        data_acc = []
        i_old = 0
        for i in per_var:
            i_old = i_old + i
            data_acc.append( i_old )
        plt.bar( x=range( 1, len( data_acc ) + 1 ), height=data_acc, tick_label=labels )
        plt.ylabel( "accumulate variance" )
        plt.xlabel( "Principle Components" )
        plt.title( "Bar plot" )
        plt.show()
        return
    def plot(self):
        my_fig = plt.figure()
        ax = plt.subplot( 111 )
        plt.gca().invert_xaxis()  # inverts values of x-axis
        ncomp = 2
        for icomp in range( ncomp ):
            ax.plot( self.my_wl , loadings[icomp], label='PC' + str( icomp + 1 ) )
        ax.set_xlabel( 'wavenumber ' )
        ax.set_ylabel( 'intensity (normalized)' )
        ax.set_yticklabels( [] )
        ax.legend()
        plt.title( 'PCA-Loadings' )
        my_fig.savefig( 'PCA-Loadings.png' )
        my_fig = plt.figure()
        ax = plt.subplot( 111 )
        ax.plot( transformed_data[0], transformed_data[1], '.' )
        ax.set_xlim( np.quantile( transformed_data[0], 0.05 ), np.quantile( transformed_data[0], 0.95 ) )
        ax.set_ylim( np.quantile( transformed_data[1], 0.05 ), np.quantile( transformed_data[1], 0.95 ) )
        ax.set_xlabel( 'PC1' )
        ax.set_ylabel( 'PC2' )
        plt.title( 'scatterplot' )
        my_fig.savefig( 'scatterplot.png' )
        my_fig.tight_layout()
        plt.show()
        return
    def export(self):
        b=len(self.my_wl)
        save_mean = np.zeros((b, 2), dtype=float)
        save_mean[:, 0] = self.my_wl
        save_mean[:, 1] = mean_spc
        save_std = np.zeros((b, 2), dtype=float)
        save_std[:, 0] = self.my_wl
        save_std[:, 1] = std_spc
        save_PC1 = np.zeros((b,2), dtype=float)
        save_PC1[:,0]=self.my_wl
        save_PC1[:,1]=loadings[0]
        save_PC2 = np.zeros((b, 2), dtype=float)
        save_PC2[:, 0] = self.my_wl
        save_PC2[:,1]=loadings[1]
        np.savetxt( "mean spectra.txt", save_mean, delimiter='\t', header='wavenumber\tmean_spc', fmt="%8.5f" )
        np.savetxt( "std spectra.txt", save_std, delimiter='\t', header='wavenumber\tstd_spc', fmt="%8.5f" )
        np.savetxt( "PCA1 Loadings.txt", save_PC1, delimiter='\t', header='wavenumber\tPC1', fmt="%8.5f" )
        np.savetxt( "PCA2 Loadings.txt", save_PC2, delimiter='\t', header='wavenumber\tPC2', fmt="%8.5f" )
        join( self.path, "mean_spc.txt" )
        join( self.path, "std_spc.txt" )

a=PCA_calculator(path)
my_spc , wl = a.my_spc , a.my_wl
print("mean_spc",my_spc.shape)
print("wl",wl.shape)
spc_norm , my_sum ,mean_spc,std_spc,transformed_data,loadings = a.pca(ncomp=2)
a.export() #save mean , std , PC1 and PC2
a.plot() #plot PCA loadings
a.pca_var(ncomp=3) #plot variance percentages




