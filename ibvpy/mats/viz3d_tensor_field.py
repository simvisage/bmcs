'''
Created on Feb 11, 2018

@author: rch
'''

import os

from mayavi import mlab
from mayavi.filters.api import ExtractTensorComponents
from mayavi.modules.api import Surface
from mayavi.sources.vtk_xml_file_reader import VTKXMLFileReader
from tvtk.api import write_data

import numpy as np
import traits.api as tr
import traitsui.api as ui

from .viz3d_field import Vis3DField, Viz3DField


class Vis3DTensorField(Vis3DField):

    var = tr.Str('eps_ab')

    def update(self):
        ts = self.sim.tstep
        U = ts.U_k
        t_n1 = ts.t_n1
        U_vector_fields = []
        var_Encd_tensor_fields = []
        for domain in ts.fe_domain:
            xdomain = domain.xdomain
            var_function = domain.tmodel.var_dict.get(self.var, None)
            if var_function == None or xdomain.hidden:
                continue
            fets = xdomain.fets
            DELTA_x_ab = fets.vtk_expand_operator
            DELTA_f_ab = xdomain.vtk_expand_operator
            U_Eia = U[xdomain.o_Eia]
            _, _, n_a = U_Eia.shape
            U_vector_fields.append(np.einsum(
                'Ia,ab->Ib', U_Eia.reshape(-1, n_a), DELTA_x_ab
            ))
            eps_Enab = xdomain.map_U_to_field(U)
            # copy the state variable as the operator
            var_Enab = var_function(eps_Enab, t_n1, **domain.state_k)
            var_Encd = np.einsum(
                '...ab,ac,bd->...cd',
                var_Enab, DELTA_f_ab, DELTA_f_ab
            )
            var_Encd_tensor_fields.append(var_Encd.reshape(-1, 9))
        self.ug.point_data.vectors = np.vstack(U_vector_fields)
        self.ug.point_data.vectors.name = 'displacement'
        self.ug.point_data.tensors = np.vstack(var_Encd_tensor_fields)
        self.ug.point_data.tensors.name = self.var
        fname = '%s_step_%08.4f' % (self.var, t_n1)
        target_file = os.path.join(
            self.dir, fname.replace('.', '_')
        ) + '.vtu'
        write_data(self.ug, target_file)
        self.add_file(target_file)


class Viz3DTensorField(Viz3DField):

    def setup(self):
        m = mlab
        fname = self.vis3d.file_list[0]
        var = self.vis3d.var
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
            data_name=var
        )
        lut.scalar_lut_manager.scalar_bar.orientation = 'horizontal'
        lut.scalar_lut_manager.scalar_bar_representation.trait_set(
            maximum_size=np.array([100000, 100000]),
            minimum_size=np.array([1, 1]),
            position=np.array([0.3, 0.05]),
            position2=np.array([0.65, 0.1]),
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

    traits_view = ui.View(
    )
