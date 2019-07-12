'''
Created on Feb 11, 2018

@author: rch
'''

import os
from mathkit.tensor import EPS

from mayavi import mlab
from tvtk.api import tvtk
from view.plot3d.viz3d import Vis3D, Viz3D

import numpy as np
import traits.api as tr


class Vis3DLattice(Vis3D):
    r'''Record the response triggered by the hist object the time increments
    upon a record event.
    '''
    var = tr.Str('<unnamed>')

    def setup(self):
        self.new_dir()
        fe_domain = self.sim.tstep.fe_domain
        X_Ia_list = []
        I_Li_list = []
        Xm_Lia_list = []
        # Consistently with other adaptors, this one allows for
        # a loop over several lattice domains.
        for domain in fe_domain:
            xdomain = domain.xdomain
            var_function = domain.tmodel.var_dict.get(self.var, None)
            if var_function == None or xdomain.hidden:
                continue
            DELTA_x_ab = xdomain.vtk_expand_operator
            X_Ia = np.einsum(
                'Ia,ab->Ib',
                xdomain.X_Ia, DELTA_x_ab
            )
            Xm_Lia = np.einsum(
                'Lib,ba->Lia',
                xdomain.Xm_Lia, DELTA_x_ab
            )
            X_Ia_list.append(X_Ia)
            Xm_Lia_list.append(Xm_Lia)
            I_Li_list.append(xdomain.I_Li)
        if len(I_Li_list) == 0:
            raise ValueError(
                'Empty output for field variable %s in model %s' %
                (self.var, domain.tmodel)
            )
        X_Ia = np.hstack(X_Ia_list)
        I_Li = np.hstack(I_Li_list)
        fname = 'lattice_tessellation'
        target_file = os.path.join(
            self.dir, fname.replace('.', '_')
        ) + '.npz'
        np.savez(target_file,
                 X_Ia=X_Ia, I_Li=I_Li, Xm_Lia=Xm_Lia)
        self.lattice_file = target_file

    def update(self):
        ts = self.sim.tstep
        U = ts.U_k
        t = ts.t_n1
        U_Ia_list = []
        U_Lipa_list = []
        state_fields = []
        for domain in ts.fe_domain:
            xdomain = domain.xdomain
            state_field = domain.state_n.get(self.var, None)
            if (xdomain.hidden):
                # If the state variable not present in the domain, skip
                continue
            if (state_field is not None):
                state_fields.append(state_field.flatten())
            DELTA_x_ab = xdomain.vtk_expand_operator
            U_Ib = U[xdomain.o_Ipa][..., 0, :]
            U_Ia_list.append(np.einsum(
                '...a,ab->...b', U_Ib, DELTA_x_ab
            ))
            U_Lipa = U[xdomain.o_Lipa]
            U_Lipa_list.append(np.einsum(
                '...a,ab->...b', U_Lipa, DELTA_x_ab
            ))
            if len(U_Lipa) == 0:
                raise ValueError('no fields for variable %s' % self.var)
        # Save the displacements
        U_Ia = np.vstack(U_Ia_list)
        U_Lipa = np.vstack(U_Lipa_list)
 #       state_vec = np.hstack(state_fields)
        fname = 'U_step_%008.4f' % t
        target_file = os.path.join(
            self.dir, fname.replace('.', '_')
        ) + '.npz'
        # , state_vec=state_vec)
        np.savez(target_file, U_Ia=U_Ia, U_Lipa=U_Lipa)
        self.add_file(target_file)


class Viz3DLattice(Viz3D):

    warp_factor = tr.Float(1.0, auto_set=False, enter_set=True)

    def setup(self):
        m = mlab

        npzfile = np.load(self.vis3d.lattice_file)
        X_Ia = npzfile['X_Ia']
        I_Li = npzfile['I_Li']
        Xm_Lia = npzfile['Xm_Lia']

        self.d = tvtk.PolyData(points=X_Ia, lines=I_Li)
        self.warp_vector = m.pipeline.warp_vector(self.d)
        self.warp_vector.filter.scale_factor = self.warp_factor

        tube = m.pipeline.tube(self.warp_vector,
                               tube_radius=self.tube_radius)
        self.lines = m.pipeline.surface(tube, color=(0.1, 0.1, 0.1))

        X_aiL = np.einsum('Lia->aiL', X_Ia[I_Li])
        Xm_aiL = np.einsum('Lia->aiL', Xm_Lia)
        XIm_aiL = np.concatenate(
            [
                np.einsum('iaL->aiL',
                          np.array([X_aiL[:, 0, ...], Xm_aiL[:, 0, ...]])),
                np.einsum('iaL->aiL',
                          np.array([X_aiL[:, 1, ...], Xm_aiL[:, 0, ...]]))
            ], axis=-1
        )
        XIm_aLi = np.einsum('aiL->aLi', XIm_aiL)
        XIm_aJ = XIm_aLi.reshape(3, -1)
        XIm_Ja = np.einsum('aJ->Ja', XIm_aJ)
        J_Li = np.arange(XIm_aJ.shape[1]).reshape(-1, 2)
        self.d_J = tvtk.PolyData(points=XIm_Ja, lines=J_Li)
        self.warp_vector_J = m.pipeline.warp_vector(self.d_J)
        self.warp_vector_J.filter.scale_factor = self.warp_factor

        tube_J = m.pipeline.tube(self.warp_vector_J,
                                 tube_radius=self.tube_radius)
        self.lines_J = m.pipeline.surface(tube_J, color=(0.1, 0.1, 0.1))
        glyph = m.pipeline.glyph(self.warp_vector_J)

        glyph.glyph.glyph_source.glyph_source = glyph.glyph.glyph_source.glyph_dict[
            'cube_source']
        glyph.glyph.glyph.range = np.array([0., 0.1])
        glyph.glyph.scale_mode = 'scale_by_vector'
        glyph.glyph.color_mode = 'color_by_vector'

    tube_radius = tr.Float(0.01)

    visible = tr.Property(tr.Bool)

    def _set_visible(self, visible):
        return
        self.d.visible = visible

    def _get_visible(self):
        return True  # self.d.visible

    def plot(self, vot):
        fe_domain = self.vis3d.sim.tstep.fe_domain
        xdomain = fe_domain[0].xdomain
        dXm_Lia = xdomain.dXm_Lia

        idx = self.vis3d.sim.hist.get_time_idx(vot)
        fname = self.vis3d.file_list[idx]
        npzfile = np.load(fname)
        U_Ia = npzfile['U_Ia']
        self.d.point_data.vectors = U_Ia

        U_Lipa = npzfile['U_Lipa']
        Um_trans_Lia, Phi_Lia = np.einsum('Lipa->pLia', U_Lipa)
        Um_rot_Lia = np.einsum('abc,...b,...c->...a',
                               EPS, Phi_Lia, dXm_Lia)
        Um_Lia = Um_trans_Lia + Um_rot_Lia
        Um_trans_aiL = np.einsum('Lia->aiL', Um_trans_Lia)
        Um_aiL = np.einsum('Lia->aiL', Um_Lia)
        UIm_aiL = np.concatenate(
            [
                np.einsum('iaL->aiL',
                          np.array([Um_trans_aiL[:, 0, ...], Um_aiL[:, 0, ...]])),
                np.einsum('iaL->aiL',
                          np.array([Um_trans_aiL[:, 1, ...], Um_aiL[:, 1, ...]]))
            ], axis=-1
        )
        UIm_aLi = np.einsum('aiL->aLi', UIm_aiL)
        UIm_aJ = UIm_aLi.reshape(3, -1)
        UIm_Ja = np.einsum('aJ->Ja', UIm_aJ)
        self.d_J.point_data.vectors = UIm_Ja
