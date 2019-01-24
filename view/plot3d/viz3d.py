'''
Created on Feb 11, 2018

@author: rch
'''

import tempfile
import traits.api as tr


class Vis3D(tr.HasTraits):

    sim = tr.WeakRef

    dir = tr.Directory

    file_list = tr.List(tr.Str,
                        desc='a list of files belonging to a time series')

    def setup(self):
        pass

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
