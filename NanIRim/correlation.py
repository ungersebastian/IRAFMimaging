'''
Runs some basic analysis on PiFM-data of Daniela Taeuber. 

Created by: René Lachmann
Date: 20191204
Last edit: 20190318
'''


# %% imports
from skimage.transform import warp_polar, rotate
from skimage.feature import register_translation
import sys
import os
from datetime import datetime
import NanoImagingPack as nip
import numpy as np
import matplotlib.pyplot as plt
import importlib as il
if False:  # to conserve order (which gets swirled up by pep)
    sys.path.append(
        "r//mars/usr/FA8_Mikroskopie/FAG82_BiomedizinischeBildgebung/BioPOLIM/PiFM/")
    import MicroPy as mipy

today = datetime.strftime(datetime.now(), "%Y%m%d")

# %% parameters -------------------------------------------------
general_path = r'//mars/usr/FA8_Mikroskopie/FAG82_BiomedizinischeBildgebung/BioPOLIM/PiFM/'
load_path = general_path + 'Bacteria/210727BacVan15/'
save_path = general_path + 'Bacteria/210727BacVan15/BacVan15Results' + today + '/'


if 0:
    il.reload(mipy)

mysel = 1
if mysel == 1:
    load_wd = ['BacVan15_0024PiFMFwd.int', 'BacVan15_0025PiFMFwd.int']#be aware of data type "PiFM" or "PiF"
    load_topo = ['BacVan15_0024TopographyFwd.int',
                 'BacVan15_0025TopographyFwd.int', ]
else:
    pass

# prior knowledge
data_type = np.dtype(np.int32)

# %% Magic commands ipython
# plot_inline = False
# if(plot_inline):
# %matplotlib inline
# else:
# %matplotlib qt

# %% functions


def load_daniela(load_path, load_wd, data_type):
    '''
    Loads data from a list using a given data_tpe
    Be careful to check formats whether to be "PiFM" or "PiF"
    '''
    load_stack = []
    for m in range(len(load_wd)):
        with open(load_path + load_wd[m], 'rb') as fopen:
            load_stack.append(fopen.read())
            news = [int(np.sqrt(len(load_stack[m])/4)), ]*2
            load_stack[m] = np.reshape(
                np.frombuffer(load_stack[m], data_type), news)
        # print("len(im_topo[0]) = {} = 1024**2\nlen(im_topo[1]) = {} = 512**2".format(len(im_topo[0]), len(im_topo[1])))
    return load_stack


def rescale_images(im1, im2, scale_direction='smaller', doPad=True):
    '''
        Rescales the to images to have the same shape. "scale_direction" marks whether the smaller or bigger image size shall be preserved.
    '''
    news = im2.shape[0] / \
        im1.shape[0] if scale_direction == 'smaller' else im1.shape[0]/im2.shape[0]
    if news > 1:
        rescale = np.repeat(1.0/news, 2, axis=0)
        im2 = nip.resample(im2, factors=rescale, dampOutside=True)
    elif news < 1:
        rescale = news
        im1 = nip.resample(im1, factors=rescale, dampOutside=True)
    return im1, im2, news

def offset_range_correct(im, offset=0, range=1):
    '''
    Corrects an image to have a certain range and offset
    '''
    im = im - im.min()
    im = im/im.max() * range
    im += offset

    return im


def find_shape_and_shift(im1, im2, show_im=False):
    '''
    compares images for different shape and shifts them accordingly. Assumes 2D images which are quadratic, for now.
    Shift will be applied to image 1, as image 2 is assumed to be the referce image.
    '''

    # rescale images
    im1, im2, news = rescale_images(im1, im2, scale_direction='smaller')

    # correct value-range
    im1 = offset_range_correct(im1)
    im2 = offset_range_correct(im2)

    # findshift
    #shift, _, _, _ = findshift(nip.DampOutside(
    #    im1), nip.DampOutside(im2), prec=100)
    shift, _, _, _ = findshift(nip.DampEdge(
        im1), nip.DampEdge(im2))

    # apply shift
    # im1sh = nip.shift2Dby(im1, shift)
    im1sh = nip.shift(im1, -shift, dampOutside=True)

    # display
    if show_im:
        nip.v5(nip.catE((im1, im1sh, im2)))
        # fig1 = plt.figure()
        # im_stack = np.transpose(nip.cat((im1sh, im2, np.zeros(im1sh.shape))), [1, 2, 0])
        # plt.imshow(im_stack)

    # return
    return im1sh, im2, shift, im1  # fig1


def crop_fromshift(im, shift):
    '''
    Crops an image according to its applied shift.

    :PARAMS:
    ========
    im:    input image
    cp:    crop with coordinate [xoffset,x,y,]
    '''

    #co = np.array([[0, 0], ]*len(shift), dtype=np.uint)
    # for m in range(len(shift)):
    #    if shift[m] >= 0:
    #        co[m] = [shift[m], im.shape[m]]
    #    else:
    #        co[m] = [0, im.shape[m]+shift[m]]
    #cp = np.array(np.floor(np.array(im.shape)/2.0), dtype=np.uint16)
    #cp += np.array(shift, dtype=np.uint16)
    co = np.array(np.array(im.shape) - np.abs(shift), dtype=np.uint8)
    cp = np.array(np.array(im.shape)/2.0-np.floor(shift), dtype=np.uint)

    return nip.extract(im, co, cp)


def get_properties(im):
    im_dat = stf_basic(im, printout=True)
    print('Image value-range = ' + str(im_dat[0] - im_dat[1]))
    return im_dat


# SOME FUNCTIONS FROM MY MIPY-Toolbox
def findshift(im1, im2, prec=100, printout=False):
    '''
    Just a wrapper for the Skimage function using sub-pixel shifts, but with nice info-printout.
    link: https://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html
    :param:
    =======
    :im1: reference image
    :im2: shifted image
    :prec: upsample-factor = number of digits used for sub-pix-precision (for sub-sampling)

    :out:
    =====
    :shift: calculated shift-vector
    :error: translation invariant normalized RMS error between images
    :diffphase: global phase between images -> should be 0
    :tend: time needed for processing
    '''
    from time import time
    from skimage.feature import register_translation
    tstart = time()
    # 'real' marks that input-data still has to be fft-ed
    shift, error, diffphase = register_translation(
        im1, im2, prec, space='real', return_error=True)
    tend = np.round(time() - tstart, 2)
    if printout:
        print("Found shifts={} with upsampling={}, error={} and diffphase={} in {}s.".format(
            shift, prec, np.round(error, 4), diffphase, tend))
    return shift, error, diffphase, tend


def stf_basic(im, printout=False):
    '''
    Basic statistical sharpness metrics: MAX,MIN,MEAN,MEDIAN,VAR,NVAR. Reducing the whole dimensionality to 1 value.
    '''
    im_res = list()
    use_axis = (-2, -1)
    im_res.append(np.max(im, axis=use_axis))
    im_res.append(np.min(im, axis=use_axis))
    im_res.append(np.mean(im, axis=use_axis))
    im_res.append(np.median(im, axis=use_axis))
    im_res.append(np.var(im, axis=use_axis))
    im_res.append(im_res[4]/im_res[2]**2)  # normalized variance (NVAR)
    if printout == True:
        print("Basic analysis yields:\nMAX=\t{}\nMIN=\t{}\nMEAN=\t{}\nMEDIAN=\t{}\nVAR=\t{}\nNVAR=\t{}".format(
            im_res[0], im_res[1], im_res[2], im_res[3], im_res[4], im_res[5]))
    return np.array(im_res)


def dir_test_existance(mydir):
    try:
        if not os.path.exists(mydir):
            os.makedirs(mydir)
            # logger.debug(
            #    'Folder: {0} created successfully'.format(mydir))
    finally:
        # logger.debug('Folder check done!')
        pass


def transpose_arbitrary(imstack, idx_startpos=[-2, -1], idx_endpos=[0, 1]):
    '''
    creates the forward- and backward transpose-list to change stride-order for easy access on elements at particular positions. 

    TODO: add security/safety checks
    '''
    # some sanity
    if type(idx_startpos) == int:
        idx_startpos = [idx_startpos, ]
    if type(idx_endpos) == int:
        idx_endpos = [idx_endpos, ]
    # create transpose list
    trlist = list(range(imstack.ndim))
    for m in range(len(idx_startpos)):
        idxh = trlist[idx_startpos[m]]
        trlist[idx_startpos[m]] = trlist[idx_endpos[m]]
        trlist[idx_endpos[m]] = idxh
    return trlist


def diff_tenengrad(im):
    '''
    Calculates Tenengrad-Sharpness Metric.
    '''
    impix = 1.0 / np.sqrt(np.prod(im.shape))
    return impix * np.sum(diff_sobel_horizontal(im)**2 + diff_sobel_vertical(im)**2, axis=(-2, -1))


def diff_sobel_horizontal(im):
    '''
    Calculates the horizontal sobel-filter.
    Filter-shape: [[-1 0 1],[ -2 0 2],[-1 0 1]] -> separabel:  np.outer(np.transpose([1,2,1]),[-1,0,1])
    '''
    # use separability
    trlist = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[1, 0])
    im = np.transpose(im, trlist)
    x_res = im[:, 2:] - im[:, :-2]  # only acts on x
    xy_res = x_res[:-2] + 2*x_res[1:-1] + x_res[2:]  # only uses the y-coords
    return np.transpose(xy_res, trlist)


def diff_sobel_vertical(im):
    '''
    Calculates the vertical sobel-filter.
    Filter-shape: [[-1,-2,-1],[0,0,0],[1,2,1]] -> separabel:  np.outer(np.transpose([-1,0,1]),[1,2,1])
    '''
    # use separability
    trlist = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[1, 0])
    im = np.transpose(im, trlist)
    x_res = im[:, :-2] + 2*im[:, 1:-1] + im[:, 2:]  # only x coords
    xy_res = x_res[2:] - x_res[:-2]  # further on y coords
    return np.transpose(xy_res)


'''
ONLINE Example for scaling and rotation
https://scikit-image.org/docs/dev/auto_examples/transform/plot_register_rotation.html#sphx-glr-auto-examples-transform-plot-register-rotation-py
'''


def find_rotation(im1, im2):
    '''
    Finds rotation between to images by finding the shift between them in polar-coordinates. MAPPING: (x,y) -> (rho,theta)
    TODO: Add stack function that searches for rotation in patches.

    :PARAMS:
    ========
    :im1,im2:   2D-input images of same size

    :OUTPUT:
    ========
    :rotation:      Rotation (in DEG) that needs to apply to rotate im1 onto im2. Hence, use it with negative sign if im2 shall be changed.
    '''
    rho_max = np.array(np.floor(im1.shape[-1]/2.0), dtype=np.uint16)
    im1_polar = warp_polar(im1, radius=rho_max, multichannel=False)
    im2_polar = warp_polar(im2, radius=rho_max, multichannel=False)
    rotation, _, _, _ = findshift(im1_polar, im2_polar)
    return rotation, im1_polar, im2_polar


def register_via_projection():
    '''
    Tries to find landmarks and then do non-local affine transformations to restore
    '''
    pass


# %% run ----------------------------
# load files
im_wd = load_daniela(load_path=load_path,
                     load_wd=load_wd, data_type=data_type)
im_topo = load_daniela(load_path=load_path,
                       load_wd=load_topo, data_type=data_type)

# establish save-path
dir_test_existance(save_path)


# shift image
ims1, ims2, shift, im1 = find_shape_and_shift(
    nip.image(im_topo[0]), nip.image(im_topo[1]), show_im=True)

# try using the marker-based ScikitImage routines for registration:
# The user has to select matching points
from skimage import transform as transf
v = nip.v5(nip.catE((im1,ims2)))
input('Please position markers in alternating elements (n to untick:n use automatic maximum finding, m sets marker, e toggle images.')
mm = v.getMarkers()
src = np.array(mm[::2])[:,4:2:-1]
dst = np.array(mm[1::2])[:,4:2:-1]
if 0:
    tform = transf.estimate_transform('similarity', src, dst)
else:
    tform = transf.estimate_transform('affine', src, dst)
im1t = nip.image(transf.warp(im1, inverse_map=tform.inverse,order=4))
nip.v5(nip.catE((im1,im1t,ims2)))

# crop images
im1st = crop_fromshift(ims1, shift)

# get properties
ims1p = get_properties(ims1)

# rotation? -> compare by eye for now
#rotation, ims1p, ims2p = find_rotation(ims1, ims2)
#nip.v5(nip.catE((nip.image(ims1p), nip.image(ims2p))))

# works like a charem
#b = nip.image(np.transpose(rotated, [2, 0, 1]))
#a = nip.rot2d(b[1], -shifts[0], padding=False)
#nip.v5(nip.catE((nip.image(image)[:, :, 1], a)))

# use edge-filters to find features -> check how it looks when using Sobel
#nip.v5(nip.catE((diff_sobel_vertical(ims1), diff_sobel_horizontal(ims1))))
