'''
Created on Feb 8, 2018

@author: rch
'''

from ibvpy.api import \
    IFETSEval, FEGrid
from ibvpy.core.i_tstepper_eval import ITStepperEval
from ibvpy.mats.mats_eval import IMATSEval
from mathkit.matrix_la import \
    SysMtxArray

import numpy as np
import traits.api as tr


@tr.provides(ITStepperEval)
class DOTSGrid(tr.HasStrictTraits):
    '''Domain time steppsr on a grid mesh
    '''
    L_x = tr.Float(200, input=True)
    L_y = tr.Float(100, input=True)
    L_z = tr.Float(100, input=True)
    n_x = tr.Int(100, input=True)
    n_y = tr.Int(30, input=True)
    n_z = tr.Int(10, input=True)
    integ_factor = tr.Float(1.0, input=True)
    fets = tr.Instance(IFETSEval, input=True)
    mats = tr.Instance(IMATSEval, input=True)
    mesh = tr.Property(tr.Instance(FEGrid), depends_on='+input')

    @tr.cached_property
    def _get_mesh(self):
        return FEGrid(coord_max=(self.L_x, self.L_y, self.L_z),
                      shape=(self.n_x, self.n_y, self.n_z),
                      fets_eval=self.fets)

    cached_grid_values = tr.Property(tr.Tuple,
                                     depends_on='+input')

    @tr.cached_property
    def _get_cached_grid_values(self):
        x_Ia = self.mesh.X_Id
        n_I, n_a = x_Ia.shape
        dof_Ia = np.arange(n_I * n_a, dtype=np.int_).reshape(n_I, -1)
        I_Ei = self.mesh.I_Ei
        x_Eia = x_Ia[I_Ei, :]
        dof_Eia = dof_Ia[I_Ei]
        x_Ema = np.einsum(
            'im,Eia->Ema', self.fets.N_im, x_Eia
        )
        J_Emar = np.einsum(
            'imr,Eia->Emar', self.fets.dN_imr, x_Eia
        )
        J_Enar = np.einsum(
            'inr,Eia->Enar', self.fets.dN_inr, x_Eia
        )
        det_J_Em = np.linalg.det(J_Emar)
        inv_J_Emar = np.linalg.inv(J_Emar)
        inv_J_Enar = np.linalg.inv(J_Enar)
        I_sym_abcd = self.fets.I_sym_abcd
        B_Eimabc = np.einsum(
            'abcd,imr,Eidr->Eimabc', I_sym_abcd, self.fets.dN_imr, inv_J_Emar
        )
        B_Einabc = np.einsum(
            'abcd,inr,Eidr->Einabc', I_sym_abcd, self.fets.dN_inr, inv_J_Enar
        )
        BB_Emicjdabef = np.einsum(
            'Eimabc,Ejmefd, Em, m->Emicjdabef', B_Eimabc, B_Eimabc,
            det_J_Em, self.fets.w_m
        )
        return (BB_Emicjdabef, B_Eimabc,
                dof_Eia, x_Eia, dof_Ia, I_Ei,
                B_Einabc, det_J_Em)

    BB_Emicjdabef = tr.Property()
    '''Quadratic form of the kinematic mapping.
    '''

    def _get_BB_Emicjdabef(self):
        return self.cached_grid_values[0]

    B_Eimabc = tr.Property()
    '''Kinematic mapping between displacements and strains in every
    integration point.
    '''

    def _get_B_Eimabc(self):
        return self.cached_grid_values[1]

    B_Einabc = tr.Property()
    '''Kinematic mapping between displacement and strain in every
    visualization point
    '''

    def _get_B_Einabc(self):
        return self.cached_grid_values[6]

    dof_Eia = tr.Property()
    '''Mapping [element, node, direction] -> degree of freedom.
    '''

    def _get_dof_Eia(self):
        return self.cached_grid_values[2]

    x_Eia = tr.Property()
    '''Mapping [element, node, direction] -> value of coordinate.
    '''

    def _get_x_Eia(self):
        return self.cached_grid_values[3]

    dof_Ia = tr.Property()
    '''[global node, direction] -> degree of freedom
    '''

    def _get_dof_Ia(self):
        return self.cached_grid_values[4]

    I_Ei = tr.Property()
    '''[element, node] -> global node
    '''

    def _get_I_Ei(self):
        return self.cached_grid_values[5]

    det_J_Em = tr.Property()
    '''Jacobi determinant in every element and integration point.
    '''

    def _get_det_J_Em(self):
        return self.cached_grid_values[7]

    state_arrays = tr.Property(tr.Dict(tr.Str, tr.Array),
                               depends_on='fets, mats')
    '''Dictionary of state arrays.
    The entry names and shapes are defined by the material
    model.
    '''
    @tr.cached_property
    def _get_state_arrays(self):
        return {
            name: np.zeros((self.mesh.n_active_elems, self.fets.n_m,)
                           + mats_sa_shape, dtype=np.float_)
            for name, mats_sa_shape
            in list(self.mats.state_array_shapes.items())
        }

    def get_corr_pred(self, U, dU, t_n, t_n1, update_state, algorithmic):
        '''Get the corrector and predictor for the given increment
        of unknown .
        '''
        n_c = self.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[self.I_Ei]
        eps_Emab = np.einsum(
            'Eimabc,Eic->Emab', self.B_Eimabc, U_Eia
        )
        dU_Ia = dU.reshape(-1, n_c)
        dU_Eia = dU_Ia[self.I_Ei]
        deps_Emab = np.einsum(
            'Eimabc,Eic->Emab', self.B_Eimabc, dU_Eia
        )
        D_Emabef, sig_Emab = self.mats.get_corr_pred(
            eps_Emab, deps_Emab, t_n, t_n1, update_state, algorithmic,
            **self.state_arrays
        )
        K_Eicjd = self.integ_factor * np.einsum(
            'Emicjdabef,Emabef->Eicjd', self.BB_Emicjdabef, D_Emabef
        )
        n_E, n_i, n_c, n_j, n_d = K_Eicjd.shape
        K_E = K_Eicjd.reshape(-1, n_i * n_c, n_j * n_d)
        dof_E = self.dof_Eia.reshape(-1, n_i * n_c)
        K_subdomain = SysMtxArray(mtx_arr=K_E, dof_map_arr=dof_E)
        f_Eic = self.integ_factor * np.einsum(
            'm,Eimabc,Emab,Em->Eic', self.fets.w_m, self.B_Eimabc, sig_Emab,
            self.det_J_Em
        )
        f_Ei = f_Eic.reshape(-1, n_i * n_c)
        F_dof = np.bincount(dof_E.flatten(), weights=f_Ei.flatten())
        F_int = F_dof
        norm_F_int = np.linalg.norm(F_int)
        return K_subdomain, F_int, norm_F_int
