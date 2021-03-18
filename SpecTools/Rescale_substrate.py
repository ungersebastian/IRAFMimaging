"""This program rescale data based on division by substrate data then ask if you need PCA calculation
Mohammad Soltaninezhad
Dr.Daniela Täuber 18 March2021 """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
PATH_substrate = "CaF20001 mean curve.txt"  # csv or txt format enter substrate data
df_sub = pd.read_csv( PATH_substrate, sep="\t",skiprows=[],delimiter="\t" )  # in skiprows write the rows which contain text in order to elliminate them
PATH_test = "200129_Ret240010_RawSpectra.txt"  # csv or txt format enter your data which need rescalling
df_test = pd.read_csv( PATH_test, sep="\t",skiprows=[],delimiter="\t")
y_sub = df_sub.values.T
y_testm = df_test.values.T
x=y_testm[0]
x = np.nan_to_num(x)
y_sub = np.nan_to_num(y_sub)
y_testm = np.nan_to_num(y_testm)
y_sub= np.delete( y_sub, 0, 0 )
y_test = np.delete( y_testm, 0, 0 )
print("wavelength shape",x.shape)
print("wavelength ",x)
print("raw spectrum shape",y_test.shape)
print("substrate spectrum shape",y_sub.shape)
b = len( y_test )
plt.plot( x, y_test.T )
plt.title( 'raw spectra' )
plt.show()
plt.plot( x, y_sub.T )
plt.title( 'substrate mean' )
plt.show()
rescale_data = np.zeros( y_test.shape )
for i in range( b ):
    rescale_data[i, :] = y_test[i, :] / y_sub
print("rescaledata",rescale_data.shape)
plt.plot( x, rescale_data.T )
plt.title( 'Rescale spectra' )
plt.show()
r=np.zeros(y_testm.shape)
print("r",r.shape)
r[0,]=x
r[1:,]=rescale_data
r=np.transpose(r)
np.savetxt("rescale data.txt", r)
ask=input("Do you need PCA calculator?if yes enter y").lower()
if ask=="y":
    rescale_data = pd.DataFrame( rescale_data )
    rescale_data = np.nan_to_num( rescale_data )
    y_test = pd.DataFrame( y_test )
    norm_rescale = StandardScaler().fit_transform( rescale_data )  # normalizing the features
    norm_test = StandardScaler().fit_transform( y_test )  # normalizing the features
    print("norm_rescale",norm_rescale.shape)
    print("norm_test",norm_test.shape)
    mean_rescale = np.mean( norm_rescale )
    mean_test = np.mean( norm_test )
    mean_rescale = np.nan_to_num( mean_rescale )
    mean_test = np.nan_to_num( mean_test )
    std_rescale = np.std( norm_rescale )
    std_test = np.std( y_test )
    std_rescale = np.nan_to_num( std_rescale )
    std_test = np.nan_to_num( std_test )
    print("mean_test",mean_test)
    print("mean_rescale",mean_rescale)
    print("std_rescale",std_rescale)
    print("std_test",std_test)
    feat_cols_res = ['feature' + str( i ) for i in range( norm_rescale.shape[1] )]
    feat_cols_test = ['feature' + str( i ) for i in range( norm_test.shape[1] )]
    norm_rescale = pd.DataFrame( norm_rescale, columns=feat_cols_res )
    norm_test = pd.DataFrame( norm_test, columns=feat_cols_test )
    ncomp = 2
    resmod = PCA( n_components=ncomp )
    testmod = PCA( n_components=ncomp )
    pca_rescale = resmod.fit_transform( norm_rescale )
    pca_test = testmod.fit_transform( norm_test )
    loadings_rescale = resmod.components_
    loadings_test = testmod.components_
    pca_Df_rescale = pd.DataFrame( data=pca_rescale, columns=['pc1res', 'pc2res'] )
    pca_Df_test = pd.DataFrame( data=pca_test, columns=['pc1test', 'pc2test'] )
    pc1res = pca_rescale[:, 0]
    pc2res = pca_rescale[:, 1]
    plt.scatter( pc1res, pc2res )
    np.savetxt("PC1_rescale.txt",pc1res)
    np.savetxt("PC2_rescale.txt",pc2res)
    plt.title( "PCA rescale", fontsize=15 )
    plt.xlabel( 'Principal Component - 1', fontsize=15 )
    plt.ylabel( 'Principal Component - 2', fontsize=15 )
    plt.show()
    pc1test = pca_test[:, 0]
    pc2test = pca_test[:, 1]
    plt.scatter( pc1test, pc2test )
    np.savetxt( "PC1_test.txt", pc1test )
    np.savetxt( "PC2_test.txt", pc2test )
    plt.title( "PCA raw", fontsize=15 )
    plt.xlabel( 'Principal Component - 1', fontsize=15 )
    plt.ylabel( 'Principal Component - 2', fontsize=15 )
    plt.show()
    my_fig = plt.figure()
    ax = plt.subplot( 111 )
    for icomp in range( ncomp ):
        ax.plot( x, loadings_rescale[icomp], label='PC' + str( icomp + 1 ) )
    plt.title( 'PCA-Loadings rescale' )
    plt.xlabel( 'wavenumber', fontsize=15 )
    plt.ylabel( 'intensity normalized', fontsize=15 )
    plt.show()
    my_fig = plt.figure()
    ax = plt.subplot( 111 )
    for icom in range( ncomp ):
        ax.plot( x, loadings_test[icom], label='PC' + str( icom + 1 ) )
    plt.title( 'PCA-Loadings test' )
    plt.xlabel( 'wavenumber', fontsize=15 )
    plt.ylabel( 'intensity normalized', fontsize=15 )
    plt.show()

