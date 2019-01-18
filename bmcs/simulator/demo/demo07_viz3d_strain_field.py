'''
Created on Feb 11, 2018

@author: rch
'''

import os

from mayavi import mlab
from mayavi.filters.api import ExtractTensorComponents
from mayavi.modules.api import Surface
from mayavi.sources.vtk_xml_file_reader import VTKXMLFileReader
from tvtk.api import \
    tvtk, write_data

import numpy as np
import traits.api as tr
from view.plot3d.viz3d import Vis3D, Viz3D


class Vis3DField(Vis3D):

    tstep = tr.WeakRef

    var = tr.Str('<unnamed>')

    def setup(self):
        self.new_dir()
        ts = self.tstep
        xdomain = ts.xdomain
        fets = xdomain.fets
        DELTA_x_ab = fets.vtk_expand_operator

        vtk_cell_type = fets.vtk_cell_class().cell_type
        n_c = fets.n_nodal_dofs
        n_E, n_i, n_a = xdomain.x_Eia.shape
        points = np.einsum(
            'Ia,ab->Ib',
            xdomain.x_Eia.reshape(-1, n_c), DELTA_x_ab
        )
        U = np.zeros_like(points)
        self.ug = tvtk.UnstructuredGrid(points=points)
        vtk_cells = (np.arange(n_E) * n_i)[:, np.newaxis] + \
            np.array(fets.vtk_cell, dtype=np.int_)[np.newaxis, :]
        self.ug.set_cells(vtk_cell_type, vtk_cells)
        self.update(U, 0)


class Vis3DStrainField(Vis3DField):

    def update(self, U, t):
        xdomain = self.tstep.xdomain
        fets = xdomain.fets
        n_c = fets.n_nodal_dofs
        DELTA_x_ab = fets.vtk_expand_operator
        DELTA_f_ab = xdomain.vtk_expand_operator
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[xdomain.I_Ei]
        U_vector_field = np.einsum(
            'Ia,ab->Ib',
            U_Eia.reshape(-1, n_c), DELTA_x_ab
        )
        self.ug.point_data.vectors = U_vector_field
        self.ug.point_data.vectors.name = 'displacement'
        eps_Enab = xdomain.map_U_to_field(U)
        eps_Encd = np.einsum(
            '...ab,ac,bd->...cd',
            eps_Enab, DELTA_f_ab, DELTA_f_ab
        )
        eps_Encd_tensor_field = eps_Encd.reshape(-1, 9)
        self.ug.point_data.tensors = eps_Encd_tensor_field
        self.ug.point_data.tensors.name = 'strain'
        fname = '%s_step_%8.4f' % (self.var, t)
        target_file = os.path.join(
            self.dir, fname.replace('.', '_')
        ) + '.vtu'
        write_data(self.ug, target_file)
        self.add_file(target_file)


class Viz3DHist(Viz3D):

    def plot(self, vot):
        idx = self.vis3d.tstep.hist.get_time_idx(vot)
        self.d.file_list = self.vis3d.file_list
        self.d.timestep = idx + 1


class Viz3DStrainField(Viz3DHist):

    label = tr.Str('<unnambed>')
    vis3d = tr.WeakRef

    def setup(self):
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
