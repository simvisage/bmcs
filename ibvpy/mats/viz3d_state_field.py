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
from ibvpy.mats.viz3d_strain_field import Vis3DField, Viz3DHist


class Vis3DStateField(Vis3DField):

    tmodel = tr.DelegatesTo('tstep')

    state_vars = tr.Property()

    def _get_state_vars(self):
        return self.tmodel.state_var_shapes

    def update(self, U, t):
        ts = self.tstep
        xdomain = ts.xdomain
        fets = xdomain.fets
        omega_field = ts.state_n[self.var]
        n_c = fets.n_nodal_dofs
        DELTA_x_ab = fets.vtk_expand_operator
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[xdomain.I_Ei]
        U_vector_field = np.einsum(
            'Ia,ab->Ib', U_Eia.reshape(-1, n_c), DELTA_x_ab
        )
        self.ug.point_data.vectors = U_vector_field
        self.ug.point_data.vectors.name = 'displacement'
        self.ug.point_data.scalars = omega_field.flatten()
        self.ug.point_data.scalars.name = self.var
        fname = '%s_step_%008.4f' % (self.var, t)
        target_file = os.path.join(
            self.dir, fname.replace('.', '_')
        ) + '.vtu'
        write_data(self.ug, target_file)
        self.add_file(target_file)


class Viz3DStateField(Viz3DHist):

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
