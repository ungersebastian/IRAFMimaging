'''baseline correction
Mohammad Soltaninezhad sinawpppp@gmail.com
Dr.Daniela Täuber
 23 FEB 2021 '''
'''program help:
raw spectra plot After run ,then you asked if you want to do whole spectrum fitting or region select mode?
1.you can set whole spc fit poly,program ask for polynomial degree for each stage 
2.if you select region select mode you have 3 choice!!!
2.1By input==1 you  can select a desire regions by set upper and lower wavelength,
!2.2By input==2 you are able to divide spc into two different regions,
and fit every region with ideal poly order,
2.3By input==3 program divide spc into 3 different regions,each can be fitted by desire poly order
 '''
#%%
#Imports and plot of original data
import peakutils
from peakutils.plot import plot as pplot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#call the file
def find_index(wavelenght_arr,number):
    for index,value in enumerate(wavelenght_arr):
        if abs(value-number)<0.0000001:
            return index
    return index
#recive data
#PATH="D:/SpecTools/RetavforAnaconda.csv" #csv or txt format
PATH="D:/SpecTools/AuPMMA_60nm_FTIRspecularReflection_H20.txt" #csv or txt format
df=pd.read_csv(PATH, sep="\t", skiprows=[0] , delimiter="\t") #in skiprows 0 refers to one text line, \t = delimited by tab
#df=pd.read_csv(PATH, sep="\t", skiprows=[0] , delimiter=",") #delimiter =n ,
data=df.values.T
y=data
x=data[0] #wavelenght
x = np.nan_to_num(x)
y = np.nan_to_num(y)
y = np.delete(y, 0, 0 )
plt.plot(x,y.T)
plt.title('raw spectral')
plt.show()
#%%
#fitting
g=int(np.size(y, 1))
l=input("Do you need to select desire regions? yes =y no =n ").lower()
if l == "n": #1. whole spectrum fitting
    b = len(y)
    t = int( input( "plz enter polynomial degree " ) )
    c = np.zeros(y.shape)
    for i in range(b):
        base = peakutils.baseline(y[i], t)  # choose order of polynomial here
        c[i] = y[i] - base
        c[c < 0] = 0
    np.savetxt('baseline corrected spectra.txt', c.T) #Sa
    fig, (ax1, ax2, ax3) = plt.subplots( 3 )
    fig.suptitle( f'FTIR poly :{t}' )
    ax1.plot( x, y.T )
    ax2.plot( x, c.T )
    ax3.plot( x, base )
    plt.title( "noise" )
    plt.savefig( 'FTIR poly.png' )
    plt.show()
    plt.plot(x,c.T)
    plt.title( "FTIR corrected" )
    plt.savefig( 'FTIR corrected.png' )
    plt.show()
elif l == "y": #2.region select mode
    f=int(input("How many regions do you need(between 2-3)enter 1 for specific region"))
    if f==1:  #2.1. desire region
        wu=float(input("plz enter upper boundry wavelenght "))
        ou = find_index(x, wu)
        wl=float(input("plz enter lower boundry wavelenght "))
        t=int(input("plz enter polynomial degree "))
        ol = find_index(x, wl)
        xi=x[ol:ou,]
        yi=y[:,ol:ou]
        b = len( yi )
        ci = np.zeros( yi.shape )
        for i in range( b ):
            basei = peakutils.baseline( yi[i], t )  # choose order of polynomial here
            ci[i] = yi[i] - basei
            ci[ci < 0] = 0
        np.savetxt( 'baseline corrected spectra.txt', ci ) #change plot title base on your poly degree
        fig, (ax1, ax2, ax3) = plt.subplots( 3 )
        fig.suptitle( f'FTIR poly3{t}' )
        ax1.plot( xi, yi.T )
        ax2.plot( xi, ci.T )
        ax3.plot( xi, basei )
        plt.title("background")
        plt.savefig( 'FTIR poly.png' )
        plt.show()
        plt.plot( x, ci.T )
        plt.title( "FTIR corrected" )
        plt.savefig( 'FTIR corrected.png' )
        plt.show()

    elif f==2: #2.2.two different region
        f1=float(input("choose boundry wavelenght"))
        o=find_index(x,f1)
        y1 = y[:,:o ]
        y2= y[:, o:]
        xd = x[:o,]
        xu = x[o:,]
        b = len( y )
        t1 = int( input( "plz enter polynomial degree for first region " ) )
        t2 = int( input( "plz enter polynomial degree for second region " ) )
        c1 = np.zeros( y1.shape )
        c2 = np.zeros( y2.shape )
        for i in range( b ):
            base1 = peakutils.baseline( y1[i], t1 ) #select first region poly order
            base2 = peakutils.baseline( y2[i], t2 )  #select second region poly order
            c1[i] = y1[i] - base1
            c2[i] = y2[i] - base2
            c1[c1 < 0] = 0
            c1[c1 < 0] = 0
        np.savetxt( 'baseline corrected spectra1.txt', c1 )
        np.savetxt( 'baseline corrected spectra2.txt', c2 )
        fig, (ax1, ax2, ax3) = plt.subplots( 3 )
        fig.suptitle( f'First region poly{t1}' ) #change title base on your poly degree
        ax1.plot( xd, y1.T )
        ax2.plot( xd, c1.T )
        ax3.plot( xd, base1 )
        plt.title( "Background" )
        plt.savefig( 'FTIR poly.png' )
        plt.show()
        plt.plot( x, c1.T )
        plt.title( "FTIR corrected" )
        plt.savefig( 'FTIR corrected.png' )
        plt.show()
        fig, (ax1, ax2, ax3) = plt.subplots( 3 )
        fig.suptitle( f'second region poly{t2}' )
        ax1.plot( xu, y2.T )
        ax2.plot( xu, c2.T )
        ax3.plot( xu, base2 )
        plt.title( "background" )
        plt.savefig( 'FTIR poly.png' )
        plt.show()
        plt.plot( x, c2.T )
        plt.title( "FTIR corrected" )
        plt.savefig( 'FTIR corrected.png' )
        plt.show()
    elif f==3: #2.3.three different regions
        fh = float(input("choose upper boundry wavelenght"))
        fl = float(input("choose lower boundry wavelenght"))
        oh = int(find_index(x,fh))
        ol = int(find_index(x,fl))
        yl = y[:, 0:ol]
        ym = y[:, ol:oh]
        yh = y[:, oh:]
        xd = x[:ol, ]
        xm=x[ol:oh,]
        xu = x[oh:, ]
        t1 = int( input( "plz enter first region polynomial degree " ) )
        t2 = int( input( "plz enter second regionpolynomial degree " ) )
        t3 = int( input( "plz enter third region polynomial degree " ) )
        b = len( y )
        cl = np.zeros( yl.shape )
        cm = np.zeros( ym.shape )
        ch = np.zeros( yh.shape )
        for i in range( b ):
            basel = peakutils.baseline(yl[i], 3)
            basem = peakutils.baseline(ym[i], 3)
            baseh = peakutils.baseline(yh[i], 3)
            cl[i] = yl[i] - basel
            cm[i] = ym[i] - basem
            ch[i] = yh[i] - baseh
            cl[cl < 0] = 0
            cm[cm < 0] = 0
            ch[ch < 0] = 0
        np.savetxt( 'baseline corrected spectra1.txt', cl )
        np.savetxt( 'baseline corrected spectra2.txt', cm )
        np.savetxt( 'baseline corrected spectra3.txt', ch )
        fig, (ax1, ax2, ax3) = plt.subplots( 3 )
        fig.suptitle( f'First region poly{t1}' )
        ax1.plot( xd, yl.T )
        ax2.plot( xd, cl.T )
        ax3.plot( xd, basel )
        plt.title( "background" )
        plt.savefig( 'FTIR poly.png' )
        plt.show()
        plt.plot( x, cl.T )
        plt.title( "FTIR corrected" )
        plt.savefig( 'FTIR corrected.png' )
        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots( 3 )
        fig.suptitle( f'second region poly{t2}' )
        ax1.plot( xm, ym.T )
        ax2.plot( xm, cm.T )
        ax3.plot( xm, basem )
        plt.title( "background" )
        plt.savefig( 'FTIR poly.png' )
        plt.show()
        plt.plot( x, cm.T )
        plt.title( "FTIR corrected" )
        plt.savefig( 'FTIR corrected.png' )
        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots( 3 )
        fig.suptitle( f'third region poly{t3}' )
        ax1.plot( xu, yh.T )
        ax2.plot( xu, ch.T )
        ax3.plot( xu, baseh )
        plt.title( "background" )
        plt.savefig( 'FTIR poly.png' )
        plt.show()
        plt.plot( x, ch.T )
        plt.title( "FTIR corrected" )
        plt.savefig( 'FTIR corrected.png' )
        plt.show()
#%% Transpose and Export corrected data
#Transpose results data set
#corrected=c.values.T


