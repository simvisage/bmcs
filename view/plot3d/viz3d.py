'''
Created on Feb 11, 2018

@author: rch
'''

import os
import tempfile

from mathkit.tensor import DELTA23_ab
from mayavi.filters.api import ExtractTensorComponents
from mayavi.modules.api import Surface
from tvtk.api import \
    tvtk

import numpy as np
import traits.api as tr


class Vis3D(tr.HasTraits):

    dir = tr.Directory

    file_list = tr.List(tr.Str,
                        desc='a list of files belonging to a time series')

    def new_dir(self):
        self.dir = tempfile.mkdtemp()

    def add_file(self, fname):
        self.file_list.append(fname)


class Viz3D(tr.HasTraits):

    label = tr.Str('<unnambed>')
    vis3d = tr.WeakRef

    def set_tloop(self, tloop):
        self.tloop = tloop

    def setup(self):
        raise NotImplementedError

    def plot(self, vot):
        raise NotImplementedError
