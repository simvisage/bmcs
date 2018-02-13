'''
Created on Feb 11, 2018

@author: rch
'''

import os

from mayavi.filters.api import ExtractTensorComponents
from mayavi.modules.api import Surface
from tvtk.api import \
    tvtk

from mathkit.tensor import DELTA23_ab
import numpy as np
import traits.api as tr


class Vis3D(tr.HasTraits):

    pass


class Viz3D(tr.HasTraits):

    label = tr.Str('<unnambed>')
    vis3d = tr.WeakRef

    def set_tloop(self, tloop):
        self.tloop = tloop

    def setup(self):

        m = self.ftv.mlab

        ts = self.tloop.ts
        cell_class = tvtk.Quad().cell_type
        n_c = ts.fets.n_nodal_dofs
        n_i = 4
        n_E, n_i, n_a = ts.x_Eia.shape
        n_Ei = n_E * n_i
        points = np.einsum(
            'Ia,ab->Ib', ts.x_Eia.reshape(-1, n_c), DELTA23_ab)
        U = np.zeros_like(points)
        eps = np.zeros((len(points), 9), dtype=np.float_)
        self.ug = tvtk.UnstructuredGrid(points=points)
        self.ug.set_cells(cell_class, np.arange(n_E * n_i).reshape(n_E, n_i))
        self.ug.point_data.vectors = U
        self.ug.point_data.vectors.name = 'displacement'
        self.ug.point_data.tensors = eps
        self.ug.point_data.tensors.name = 'strain'
        self.src = m.pipeline.add_dataset(self.ug)
        self.warp_vector = m.pipeline.warp_vector(self.src)
        self.surf = m.pipeline.surface(self.warp_vector)

        engine = m.get_engine()
        etc = ExtractTensorComponents()
        engine.add_filter(etc, self.warp_vector)
        surface2 = Surface()
        engine.add_filter(surface2, etc)
        etc.filter.scalar_mode = 'component'
        lut = etc.children[0]
        lut.scalar_lut_manager.show_scalar_bar = True
        lut.scalar_lut_manager.show_legend = True
        lut.scalar_lut_manager.scalar_bar.height = 0.8
        lut.scalar_lut_manager.scalar_bar.width = 0.04
        lut.scalar_lut_manager.scalar_bar.position = np.array([0.82,  0.1])
        print 'FINISHED SET UP VIZ'

    def plot(self, vot):
        print 'PLOTTING', vot
        ts = self.tloop.ts
        n_c = ts.fets.n_nodal_dofs
        U = np.copy(self.tloop.U_record[-1])
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[ts.I_Ei]
        U_vector_field = np.einsum(
            'Ia,ab->Ib', U_Eia.reshape(-1, n_c), DELTA23_ab
        )

        eps_Enab = np.einsum(
            'Einabc,Eic->Enab', ts.B_Einabc, U_Eia
        )
#         sig_Enab = np.einsum(
#             'abef,Emef->Emab', self.ts.mats.D_abcd, eps_Enab
#         )
        self.ug.point_data.vectors = U_vector_field
        self.ug.point_data.vectors.name = 'displacement'
        eps_Encd = np.einsum(
            '...ab,ac,bd->...cd', eps_Enab, DELTA23_ab, DELTA23_ab
        )
        eps_Encd_tensor_field = eps_Encd.reshape(-1, 9)
        self.ug.point_data.tensors = eps_Encd_tensor_field
        self.ug.point_data.tensors.name = 'strain'
#         fname = os.path.join(self.write_dir, 'step_%008.4f' % t)
#         write_data(self.ug, fname.replace('.', '_'))
        self.src.update()
