'''
Created on Feb 11, 2018

@author: rch
'''

import os
import threading

from mayavi import mlab
from mayavi.filters.api import ExtractTensorComponents
from mayavi.modules.api import Surface
from mayavi.sources.vtk_xml_file_reader import VTKXMLFileReader
from pyface.api import GUI
from tvtk.api import \
    tvtk, write_data

from mathkit.tensor import DELTA23_ab
import numpy as np
import traits.api as tr
from viz3d import Vis3D, Viz3D


class Vis3DPoll(Vis3D):

    def setup(self, tloop):
        #self.lock = threading.Lock()
        self.tloop = tloop
        ts = self.tloop.ts
        cell_class = tvtk.Quad().cell_type
        n_c = ts.fets.n_nodal_dofs
        n_i = 4
        n_E, n_i, n_a = ts.x_Eia.shape
        n_Ei = n_E * n_i
        points = np.einsum(
            'Ia,ab->Ib', ts.x_Eia.reshape(-1, n_c), DELTA23_ab)
        U = np.zeros_like(points)
        self.ug = tvtk.UnstructuredGrid(points=points)
        self.ug.set_cells(cell_class, np.arange(n_E * n_i).reshape(n_E, n_i))
        self.update(U, 0)

    file_list = tr.List(tr.Str,
                        desc='a list of files belonging to a time series')

    def update(self, U, t):
        # self.lock.acquire()
        ts = self.tloop.ts
        n_c = ts.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[ts.I_Ei]
        U_vector_field = np.einsum(
            'Ia,ab->Ib', U_Eia.reshape(-1, n_c), DELTA23_ab
        )
        eps_Enab = np.einsum(
            'Einabc,Eic->Enab', ts.B_Einabc, U_Eia
        )
        self.ug.point_data.vectors = U_vector_field
        self.ug.point_data.vectors.name = 'displacement'
        eps_Encd = np.einsum(
            '...ab,ac,bd->...cd', eps_Enab, DELTA23_ab, DELTA23_ab
        )
        eps_Encd_tensor_field = eps_Encd.reshape(-1, 9)
        self.ug.point_data.tensors = eps_Encd_tensor_field
        self.ug.point_data.tensors.name = 'strain'
        fname = 'step_%008.4f' % t
        home = os.path.expanduser('~')
        target_file = os.path.join(
            home, 'simdb', 'simdata', fname.replace('.', '_')
        ) + '.vtu'
        print 'writing', target_file
        write_data(self.ug, target_file)
        self.file_list.append(target_file)
        print 'FILE WRITTEN'
        # self.lock.release()


class Viz3DPoll(Viz3D):

    label = tr.Str('<unnambed>')
    vis3d = tr.WeakRef

    def setup(self):
        #self.lock = threading.Lock()
        print 'SETTING UP VIZ3D'
        m = mlab
        fname = self.vis3d.file_list[0]
        self.d = VTKXMLFileReader()
        self.d.initialize(fname)
        self.src = m.pipeline.add_dataset(self.d)
        self.warp_vector = m.pipeline.warp_vector(self.src)
        self.surf = m.pipeline.surface(self.warp_vector)
        engine = m.get_engine()
        etc = ExtractTensorComponents()
        engine.add_filter(etc, self.warp_vector)
        surface2 = Surface()
        engine.add_filter(surface2, etc)
        etc.filter.scalar_mode = 'component'
        lut = etc.children[0]
        lut.scalar_lut_manager.set(
            show_scalar_bar=True,
            show_legend=True,
            data_name='strain field'
        )
        lut.scalar_lut_manager.scalar_bar.set(
            height=0.1,
            width=0.7,
            position=np.array([0.1,  0.1])
        )

#        lut.scalar_lut_manager.scalar_bar_representation.set(
#            position2=np.array([0.8,  0.1]),
#            position=np.array([0.1,  0.1]),
#            orientation=0,
#            maximum_size=np.array([100000, 100000]),
#            minimum_size=np.array([1, 1]),
#        )

    def plot(self, vot):
        idx = self.vis3d.tloop.get_time_idx(vot)
        self.d.file_list = self.vis3d.file_list
        self.d.timestep = idx + 1
