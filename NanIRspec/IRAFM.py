# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 07:50:16 2020

@author: ungersebastian
"""

# imports

from os.path import isfile, join, splitext
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import resource_filename, resource_listdir


class IRAFM(dict):

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return IRAFM(*args, **kwargs).__finalize__(self)
        return _c

    def __init__(self, *args, **kwargs):

        # unpacking of args and kwargs:
        i_args = 0
        n_args = len(args)
        use_example = 0

        path = ''

        if 'path' in kwargs:
            path = kwargs.pop('path')
        elif n_args > i_args:
            path = args[i_args]
            i_args = i_args+1
        else:
            use_example = 1

        if use_example == 0:
            if 'headerfile' in kwargs:
                headerfile = kwargs.pop('headerfile')
            elif n_args > i_args:
                headerfile = args[i_args]
                i_args = i_args+1
            else:
                use_example = 1

        if use_example == 1:
            print('Parameter unsufficient. Using example data.')
            path = resource_filename("IRAFM", "resources")
            headerfile = 'Ret29r20006.txt'

        self._data_type_ = np.dtype(np.int32)

        file_list = np.array([(
            f,
            join(path, f),
            splitext(f)[1]
        ) for f in listdir(path) if isfile(join(path, f))])

        self.name = splitext(headerfile)[0]

        path_file = file_list[file_list[:, 0] == headerfile][0, 1]

        # with io.open(path_file, 'r', encoding='utf-8') as fopen:
        with open(path_file, 'r', encoding='latin-1') as fopen:
            header_list = np.array(fopen.readlines())

        # extract the file list and supporting information
        # be careful: could also be named 'FileDesc2Begin'
        #where = np.where(header_list == 'FileDescBegin\n')[0]
        where = np.where([('FileDesc' in hl) and ('Begin\n' in hl) for hl in header_list])[0]
        where = np.append(where, len(header_list))
        files = [header_list[where[i]:where[i+1]] for i in range(len(where)-1)]
        files[-1] = files[-1][0:np.where([('FileDesc' in hl) and ('End\n' in hl)
                                         for hl in header_list])[0][0]+1]

        files = [f[(np.where([('FileDesc' in hl) and ('Begin\n' in hl) for hl in f])[0][0]+1):(np.where([('FileDesc' in hl) and ('End\n' in hl) for hl in f])[0][0])] for f in files]

        header_list = header_list[0:where[0]-1]

        del(where)

        self._init_dict_(header_list)

        del(header_list)

        files = [self._return_dict_(f) for f in files]
        files = [f for f in files if f != {}]

        # get the wavelength axis and the other main informations

        if 'FileNameWavelengths' in self.keys():
            path_wavelengths = self['FileNameWavelengths']
        else:
            path_wavelengths = ''
            for f in files:
                if 'FileNameWavelengths' in f and 'PhysUnitWavelengths' in f:
                    path_wavelengths = f['FileNameWavelengths']
                    break

        path_wavelengths = file_list[file_list[:, 0] == path_wavelengths][0, 1]

        with open(path_wavelengths, 'r') as fopen:
            wavelength = fopen.readlines()

        wavelength = [''.join(l.split('\n')) for l in wavelength][1:]
        wavelength = (np.array([l.split('\t') for l in wavelength]).T).astype(float)

        self.add('wavelength', wavelength[0])
        self.add('attenuation', wavelength[1])

        del(wavelength, path_wavelengths)

        # extract the file information

        self.add('files', files)

        del(files)

        # read the images

        for my_file in self['files']:
            path_file = join(path, my_file['FileName'])
            with open(path_file, 'rb') as fopen:
                my_im = fopen.read()
                my_im = np.frombuffer(my_im, self._data_type_)
                my_dim = int(len(my_im)/(self['xPixel']*self['yPixel']))
                if my_dim == 1:
                    news = (self['xPixel'], self['yPixel'])
                else:
                    news = (self['xPixel'], self['yPixel'], my_dim)
                my_im = np.reshape(my_im, news)

            scale = my_file['Scale']
            my_file['data'] = my_im*scale

    def add(self, key, val):
        self[key] = val

    def return_spc(self):
        pos = [my_file['Caption'] == 'hyPIRFwd' for my_file in self['files']]
        hyPIRFwd = np.array(self['files'])[pos][0]
        data = np.reshape(hyPIRFwd['data'], (hyPIRFwd['data'].shape[0] *
                          hyPIRFwd['data'].shape[1], hyPIRFwd['data'].shape[2]))
        return data

    def _return_value_(self, v):
        try:
            v = int(v)
        except:
            try:
                v = float(v)
            except:
                pass
        return v

    def _init_dict_(self, arr):
        arr = arr[[':' in l for l in arr]]
        arr = [''.join(l.split()) for l in arr]
        arr = [''.join(l.split('\n')) for l in arr]

        for l in arr:
            self.add(l.split(':', 1)[0], self._return_value_(l.split(':', 1)[1]))

    def _cleanData_(self, s):
        s = s.split(':')
        s0 = s[0]
        s1 = s[1]
        if ''.join(s0.split()) == 'FileName':
            if len(s1.split()) > 1:
                return False
        return True

    def _return_dict_(self, arr):

        arr = arr[[':' in l for l in arr]]

        check = [self._cleanData_(l) for l in arr]
        if False in check:
            return {}
        else:
            arr = [''.join(l.split()) for l in arr]
            arr = [''.join(l.split('\n')) for l in arr]
            return {l.split(':', 1)[0]: self._return_value_(l.split(':', 1)[1]) for l in arr}

    def extent(self):
        dpx = self['XScanRange']/self['xPixel']
        xlim = (0, dpx*(self['xPixel']-1))

        dpy = self['YScanRange']/self['yPixel']
        ylim = (0, dpy*(self['yPixel']-1))

        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        return extent

    def plot_all(self):
        extent = self.extent()

        plt.cmap = 'ocean'

        for my_file in self['files']:

            my_fig = plt.figure()
            ax = plt.subplot(111)
            plt.cmap = 'ocean'

            data = my_file['data']
            if data.ndim > 2:
                plt.colorbar(ax.imshow(np.sum(data, axis=2), cmap='inferno',
                             extent=extent), label=my_file['PhysUnit'])
            else:
                plt.colorbar(ax.imshow(data, extent=extent),
                             cmap='inferno', label=my_file['PhysUnit'])

            ax.set_xlabel('x scan ['+self['XPhysUnit']+']')
            ax.set_ylabel('y scan ['+self['YPhysUnit']+']')
            plt.title(my_file['Caption'])

            my_fig.tight_layout()
