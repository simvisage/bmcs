'''
Created on Feb 11, 2018

@author: rch
'''

import os

from ibvpy.mats.viz3d_field import Vis3DField, Viz3DField
from mayavi import mlab
from mayavi.sources.vtk_xml_file_reader import VTKXMLFileReader
from tvtk.api import write_data

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
        if len(U_vector_fields) == 0:
            raise ValueError('no fields for variable %s' % self.var)
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


class Viz3DScalarField(Viz3DField):

    warp_factor = tr.Float(1.0, auto_set=False, enter_set=True)

    lut_manager = tr.Property

    def _get_lut_manager(self):
        lut = self.warp_vector.children[0]
        return lut.scalar_lut_manager

    def setup(self):
        m = mlab
        fname = self.vis3d.file_list[0]
        var = self.vis3d.var
        self.d = VTKXMLFileReader()
        self.d.initialize(fname)
        self.src = m.pipeline.add_dataset(self.d)
        self.warp_vector = m.pipeline.warp_vector(self.src)
        self.warp_vector.filter.scale_factor = self.warp_factor
        self.surf = m.pipeline.surface(self.warp_vector)
        lut = self.warp_vector.children[0]
        lut.scalar_lut_manager.trait_set(
            lut_mode='Reds',
            show_scalar_bar=True,
            show_legend=True,
            data_name=var,
            use_default_range=False,
            data_range=np.array([0, 1], dtype=np.float_),
        )

        lut.scalar_lut_manager.scalar_bar.width = 0.5
        lut.scalar_lut_manager.scalar_bar.height = 0.15
        lut.scalar_lut_manager.scalar_bar.orientation = 'horizontal'
        lut.scalar_lut_manager.scalar_bar_representation.trait_set(
            maximum_size=np.array([100000, 100000]),
            minimum_size=np.array([1, 1]),
            position=np.array([0.5, 0.05]),
            position2=np.array([0.45, 0.1]),
        )
        lut.scalar_lut_manager.label_text_property.trait_set(
            font_family='times',
            italic=False,
            bold=False
        )
        lut.scalar_lut_manager.title_text_property.trait_set(
            font_family='times',
            italic=False,
            bold=False
        )
