# -*- coding: utf-8 -*-
# =============================================================================================== #
# ===          C O N F I D E N T I A L  ---  D R. D A N I E L A                               === #
# =============================================================================================== #
#                                                                                               = #
#                         Project Name : IPHT                                                   = #
#                                                                                               = #
#                     File Name : correlationTopoImage.py                                       = #
#                                                                                               = #
#                        Programmer : René Lachmann                                             = #
#                                                                                               = #
#                         Start Date : 08/10/2020                                               = #
#                                                                                               = #
#                   Last Update : October 08, 2020 [LM]                                         = #
#                                                                                               = #
# =============================================================================================== #
# Class:                                                                                        = #
#    Correlation -- get relative shift between topo images                                      = #
# =============================================================================================== #

import os
import sys
import numpy as np
import importlib as il
from datetime import datetime
import NanoImagingPack as nip
import matplotlib.pyplot as plt
from skimage.transform import warp_polar
from skimage.feature import register_translation


class Correlation:
    def __init__(self):
        # set general path of our image
        self.general_path = r"C:/Users/15025/Desktop/IPHT/Program/program(ProvidedByDaniela)/irafmimaging-master/IRAFim/Tests/"
        # set path of common file
        self.file_name = "191127_NP_correlation/"
        # the path of files
        self.path_file = self.general_path + self.file_name
        # set name of PiFM image in forward scanning case
        self.load_PiFM = ['Per-PDAGA-NP0009PiFFwd.int', 'Per-PDAGA-NP0010PiFFwd.int']
        # set name of Topo image in forward scanning case
        self.load_topo = ['Per-PDAGA-NP0009Topo OutFwd.int','Per-PDAGA-NP0010Topo OutFwd.int']
        # set the data format, later use it to set format of reading data
        self.data_type = np.dtype(np.int32)
        
        
    def main(self):
        # load files
        im_wd = self.loadDaniela(load_path=self.path_file,
                      load_wd=self.load_PiFM, data_type=self.data_type)
        im_topo = self.loadDaniela(load_path=self.path_file,
                       load_wd=self.load_topo, data_type=self.data_type)
        
        # find shift and shift image
        # im_topo[1]: the image need to be shifted
        # im_topo[0]: the reference image
        # ims1: the image after shifted
        # ims2: the reference image, no change
        # shift: the shift which should be done to im1
        # im1: the image 1 before shift, original image
        # each time we could compare two images
        ims1, ims2, shift, im1 = self.findShapeAndShift(
            nip.image(im_topo[1]), nip.image(im_topo[0]), show_im=True)
        
        # we need to get corresponding shift to the reference topo image here.
        print(shift)    # [-17.83  -0.49]
        
        
        # we could call self.findShapeAndShift again here to find shift between other images.
        
        # try using the marker-based ScikitImage routines for registration:
        # The user has to select matching points by hands
        # this method is not reproducible, so we could try, but not a good choice
        # so normally we annotate this
# =============================================================================
#         from skimage import transform as transf
#         
#         v = nip.v5(nip.catE((im1,ims2)))
#         input('Please position markers in alternating elements (n to untick:n use automatic maximum finding, m sets marker, e toggle images.')
#         mm = v.getMarkers()
#         src = np.array(mm[::2])[:,4:2:-1]
#         dst = np.array(mm[1::2])[:,4:2:-1]
#         if 0:
#             tform = transf.estimate_transform('similarity', src, dst)
#         else:
#             tform = transf.estimate_transform('affine', src, dst)
#         im1t = nip.image(transf.warp(im1, inverse_map=tform.inverse,order=4))
#         
#         nip.v5(nip.catE((im1,im1t,ims2)))
# =============================================================================
        
        # crop images
        #im1st = crop_fromshift(ims1, shift)
        
        # get properties
        # ims1p = self.getProperties(ims1)
        
        # rotation? -> compare by eye for now
        # rotation, ims1p, ims2p = self.findRotation(ims1, ims2)
        # nip.v5(nip.catE((nip.image(ims1p), nip.image(ims2p))))
        
        # works like a charem
        #b = nip.image(np.transpose(rotated, [2, 0, 1]))
        #a = nip.rot2d(b[1], -shifts[0], padding=False)
        #nip.v5(nip.catE((nip.image(image)[:, :, 1], a)))
        
        # use edge-filters to find features -> check how it looks when using Sobel
        # nip.v5(nip.catE((diff_sobel_vertical(ims1), diff_sobel_horizontal(ims1))))
    
    
    @staticmethod
    def loadDaniela(load_path, load_wd, data_type):
        '''
        Loads data from a list using a given data_type
        '''
        # create a string list
        load_stack = []
        # do the number of elements in load_wd times for-loop
        for num in range(len(load_wd)):
            # read file with byte mode
            with open(load_path + load_wd[num], 'rb') as f:
                load_stack.append(f.read())
                # devide 4 because each item in load_stack has 4 characters.
                # like \x00
                news = [int(np.sqrt(len(load_stack[num]) / 4)),] * 2
                load_stack[num] = np.reshape(np.frombuffer(load_stack[num], data_type), news)
            # print("len(im_topo[0]) = {} = 1024**2\nlen(im_topo[1]) = {} = 512**2".format(len(im_topo[0]), len(im_topo[1])))
        return load_stack

    
    @staticmethod 
    def rescaleImages(im1, im2, scale_direction="smaller", doPad=True):
        '''
            Rescales the to images to have the same shape. "scale_direction" 
            marks whether the smaller or bigger image size shall be preserved.
        '''
        # change the ratio order according to the scale_direction
        # default scale_direction is "smaller"
        if scale_direction == "smaller":
            news = im2.shape[0] / im1.shape[0]
        else:
            news = im1.shape[0] / im2.shape[0]
        
        # we know the size of image is quadratic， it means image has same rows
        # and columns, if the size of im1 is bigger than im2
        if news > 1:
            # news should be a float number, and this will repeat 1 / news 2 
            # times and get a flatten array with 2 same elements.
            rescale = np.repeat(1.0 / news, 2, axis = 0)
            # ROI: region of interest 感兴趣区域
            # parameter dstsize is none. 
            # change the size of im2 from origin to image.shape * factors
            im2 = nip.resample(im2, factors=rescale, dampOutside=True)
        # if the size of im2 is bigger than im1
        elif news < 1:
            rescale = news
            im1 = nip.resample(im1, factors=rescale, dampOutside=True)
        return im1, im2, news
    
    
    @staticmethod 
    def offsetRangeCorrect(im, offset=0, _range=1):
        '''
        Corrects an image to have a certain range and offset
        '''
        # make each element cut off the minimum value
        im = im - im.min()
        im = im/im.max() * _range
        im += offset
        
        return im
    
    
    @staticmethod 
    def findShapeAndShift(im1, im2, show_im=False):
        '''
        compares images for different shape and shifts them accordingly. 
        Assumes 2D images which are quadratic, for now.
        Shift will be applied to image 1, as image 2 is assumed to be 
        the referce image.
        paras explanation:
            im1: image1.
            im2: image2.
            show_im: a flag determine whether show image or not.
        '''
        # rescale images
        im1, im2, news = Correlation.rescaleImages(im1, im2, scale_direction='smaller')
    
        # correct value-range
        im1 = Correlation.offsetRangeCorrect(im1)
        im2 = Correlation.offsetRangeCorrect(im2)
    
        # findshift
        #shift, _, _, _ = findshift(nip.DampOutside(
        #    im1), nip.DampOutside(im2), prec=100)
        shift, _, _, _ = Correlation.findShift(nip.DampEdge(
            im1), nip.DampEdge(im2))
        # here, we only need the shift parameter and leave out other parameters
    
        # apply shift
        # im1sh = nip.shift2Dby(im1, shift)
        im1sh = nip.shift(im1, -shift, dampOutside=True)
        # shift im1 up shift unit
    
        # display
        if show_im:
            # print(im1.shape)
            # print(im1sh.shape)  # (256, 256)
            # print(im2.shape)
            # nip.v5(nip.catE((im1, im1sh, im2)))
            # print(im1sh.shape)  # (256, 256)
            # print(im1sh[0, 0])  # 0
            # print(im1sh[0, :])  # all zero value
            # fig1 = plt.figure()
            # im_stack = np.transpose(nip.cat((im1sh, im2, np.zeros(im1sh.shape))), [1, 2, 0])
            # plt.imshow(im_stack)
            pass
    
        # return
        return im1sh, im2, shift, im1  # fig1
    
    
    @staticmethod 
    def cropFromshift(im, shift):
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
    
    
    @staticmethod 
    def getProperties(im):
        im_dat = Correlation.stfBasic(im, printout=True)
        print('Image value-range = ' + str(im_dat[0] - im_dat[1]))
        return im_dat
    
    
    @staticmethod 
    def findShift(im1, im2, prec=100, printout=False):
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
        # round the time to the 2 decimal places
        # here the default value of printout is False
        tend = np.round(time() - tstart, 2)
        if printout:
            print("Found shifts={} with upsampling={}, error={} and diffphase={} in {}s.".format(
                shift, prec, np.round(error, 4), diffphase, tend))
        # directly return the shift, errorm diffphase and needed time for 
        # register_translation
        return shift, error, diffphase, tend

    
    @staticmethod 
    def stfBasic(im, printout=False):
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
    
    
    @staticmethod 
    def dirTestExistance(mydir):
        try:
            if not os.path.exists(mydir):
                os.makedirs(mydir)
                # logger.debug(
                #    'Folder: {0} created successfully'.format(mydir))
        finally:
            # logger.debug('Folder check done!')
            pass
    
    
    @staticmethod 
    def transposeArbitrary(imstack, idx_startpos=[-2, -1], idx_endpos=[0, 1]):
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
    
    
    @staticmethod 
    def diffTenengrad(im):
        '''
        Calculates Tenengrad-Sharpness Metric.
        '''
        impix = 1.0 / np.sqrt(np.prod(im.shape))
        return impix * np.sum(diff_sobel_horizontal(im)**2 + diff_sobel_vertical(im)**2, axis=(-2, -1))
    
    
    @staticmethod 
    def diffSobelHorizontal(im):
        '''
        Calculates the horizontal sobel-filter.
        Filter-shape: [[-1 0 1],[ -2 0 2],[-1 0 1]] -> separabel:  np.outer(np.transpose([1,2,1]),[-1,0,1])
        '''
        # use separability
        trlist = transposeArbitrary(im, idx_startpos=[-2, -1], idx_endpos=[1, 0])
        im = np.transpose(im, trlist)
        x_res = im[:, 2:] - im[:, :-2]  # only acts on x
        xy_res = x_res[:-2] + 2*x_res[1:-1] + x_res[2:]  # only uses the y-coords
        return np.transpose(xy_res, trlist)
    
    
    @staticmethod 
    def diffSobelVertical(im):
        '''
        Calculates the vertical sobel-filter.
        Filter-shape: [[-1,-2,-1],[0,0,0],[1,2,1]] -> separabel:  np.outer(np.transpose([-1,0,1]),[1,2,1])
        '''
        # use separability
        trlist = transposeArbitrary(im, idx_startpos=[-2, -1], idx_endpos=[1, 0])
        im = np.transpose(im, trlist)
        x_res = im[:, :-2] + 2*im[:, 1:-1] + im[:, 2:]  # only x coords
        xy_res = x_res[2:] - x_res[:-2]  # further on y coords
        return np.transpose(xy_res)
    
    
    '''
    ONLINE Example for scaling and rotation
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_register_rotation.html#sphx-glr-auto-examples-transform-plot-register-rotation-py
    '''
    
    
    @staticmethod 
    def findRotation(im1, im2):
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
        rotation, _, _, _ = findShift(im1_polar, im2_polar)
        return rotation, im1_polar, im2_polar
    
    
    @staticmethod 
    def registerViaProjection():
        '''
        Tries to find landmarks and then do non-local affine transformations to restore
        '''
        pass
    
    
if __name__ == "__main__":  
    correlation = Correlation()
    correlation.main()