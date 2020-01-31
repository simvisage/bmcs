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
from tvtk.api import \
    tvtk, write_data
from view.plot3d.viz3d import Vis3D, Viz3D

import numpy as np
import traits.api as tr


class Vis3DSDamage(Vis3D):

    def setup(self, tloop):
        self.new_dir()
        self.tloop = tloop
        fets = tloop.ts.fets
        ts = self.tloop.ts
        DELTA_ab = fets.vtk_expand_operator

        vtk_cell_type = fets.vtk_cell_class().cell_type
        n_c = fets.n_nodal_dofs
        n_i = fets.n_dof_r
        n_E, n_i, n_a = ts.x_Eia.shape
        n_Ei = n_E * n_i
        points = np.einsum(
            'Ia,ab->Ib', ts.x_Eia.reshape(-1, n_c), DELTA_ab)
        U = np.zeros_like(points)
        self.ug = tvtk.UnstructuredGrid(points=points)
        vtk_cells = (np.arange(n_E) * n_i)[:, np.newaxis] + \
            np.array(fets.vtk_cell, dtype=np.int_)[np.newaxis, :]
        self.ug.set_cells(vtk_cell_type, vtk_cells)
        self.update(U, 0)

    file_list = tr.List(tr.Str,
                        desc='a list of files belonging to a time series')

    def update(self, U, t):
        ts = self.tloop.ts
        fets = ts.fets
        omega_field = ts.state_arrays['omega']
        n_c = fets.n_nodal_dofs
        DELTA_ab = fets.vtk_expand_operator
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[ts.I_Ei]
        U_vector_field = np.einsum(
            'Ia,ab->Ib', U_Eia.reshape(-1, n_c), DELTA_ab
        )
        self.ug.point_data.vectors = U_vector_field
        self.ug.point_data.vectors.name = 'displacement'
        self.ug.point_data.scalars = omega_field.flatten()
        self.ug.point_data.scalars.name = 'damage'
        fname = 'omega_step_%008.4f' % t
        target_file = os.path.join(
            self.dir, fname.replace('.', '_')
        ) + '.vtu'
        write_data(self.ug, target_file)
        self.add_file(target_file)


class Viz3DSDamage(Viz3D):

    label = tr.Str('damage')
    vis3d = tr.WeakRef

    warp_factor = tr.Float(1.0, auto_set=False, enter_set=True)

    def setup(self):
        m = mlab
        fname = self.vis3d.file_list[0]
        self.d = VTKXMLFileReader()
        self.d.initialize(fname)
        self.src = m.pipeline.add_dataset(self.d)
        self.warp_vector = m.pipeline.warp_vector(self.src)
        self.warp_vector.filter.scale_factor = self.warp_factor
        self.surf = m.pipeline.surface(self.warp_vector)
        lut = self.warp_vector.children[0]
        lut.scalar_lut_manager.set(
            lut_mode='Reds',
            show_scalar_bar=True,
            show_legend=True,
            data_name='damage',
            use_default_range=False,
            data_range=np.array([0, 1], dtype=np.float_)
        )

    def plot(self, vot):
        idx = self.vis3d.tloop.get_time_idx(vot)
        self.d.file_list = self.vis3d.file_list
        self.d.timestep = idx + 1
