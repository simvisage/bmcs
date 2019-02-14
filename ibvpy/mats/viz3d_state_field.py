'''
Created on Feb 11, 2018

@author: rch
'''

import os

from mayavi import mlab
from mayavi.sources.vtk_xml_file_reader import VTKXMLFileReader
from tvtk.api import write_data

from ibvpy.mats.viz3d_field import Vis3DField, Viz3DField
import numpy as np
import traits.api as tr


class Vis3DStateField(Vis3DField):

    model = tr.DelegatesTo('sim')

    state_vars = tr.Property()

    def _get_state_vars(self):
        return self.model.state_var_shapes

    def update(self):
        ts = self.sim.tstep
        U = ts.U_k
        t = ts.t_n1
        # this needs to be adopted to tstep[domain_state]
        # loop over the subdomains
        U_vector_fields = []
        state_fields = []
        for domain in ts.fe_domain:
            xdomain = domain.xdomain
            fets = xdomain.fets
            state_field = domain.state_n.get(self.var, None)
            if (xdomain.hidden) or (state_field is None):
                # If the state variable not present in the domain, skip
                continue
            state_fields.append(state_field.flatten())
            DELTA_x_ab = fets.vtk_expand_operator
            U_Eia = U[xdomain.o_Eia]
            _, _, n_a = U_Eia.shape
            U_vector_fields.append(np.einsum(
                'Ia,ab->Ib', U_Eia.reshape(-1, n_a), DELTA_x_ab
            ))
        self.ug.point_data.vectors = np.vstack(U_vector_fields)
        self.ug.point_data.vectors.name = 'displacement'
        self.ug.point_data.scalars = np.hstack(state_fields)
        self.ug.point_data.scalars.name = self.var
        fname = '%s_step_%008.4f' % (self.var, t)
        target_file = os.path.join(
            self.dir, fname.replace('.', '_')
        ) + '.vtu'
        write_data(self.ug, target_file)
        self.add_file(target_file)


class Viz3DStateField(Viz3DField):

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
