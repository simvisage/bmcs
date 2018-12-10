
from math import fabs, pi as Pi, sqrt
from numpy import array, zeros, float_
from traits.api import \
    Array, Float
from ibvpy.api import \
    BCSlice, FETSEval
from ibvpy.fets.fets_eval import \
    RTraceEvalElemFieldVar
from ibvpy.mats.mats1D import \
    MATS1DPlastic, MATS1DElastic
from ibvpy.mats.mats1D5.mats1D5_bond import \
    MATS1D5Bond


class FETS1D52L8ULRH(FETSEval):
    '''
    Fe Bar 3 nodes, deformation
    '''

    # Dimensional mapping
    dim_slice = slice(0, 2)
    n_e_dofs = 16
    n_nodal_dofs = 2
    ngp_r = 4

    # todo - should the generic fets_eval include multipliers?
    #
    A_phase_1 = Float(1.0, desc='Cross sectional area of phase 1')
    A_phase_2 = Float(1.0, desc='Cross sectional area of phase 2')

    # Node position distribution
    dof_r = Array(value=[[-1, -1],
                         [1, -1],
                         [1, 1],
                         [-1, 1],
                         [-1. / 3., -1],
                         [1. / 3., -1],
                         [1. / 3., 1],
                         [-1. / 3., 1]])
    geo_r = Array(value=[[-1, -1],
                         [1, -1],
                         [1, 1],
                         [-1, 1]])
    vtk_r = Array(value=[[-1, -1],
                         [1, -1],
                         [1, 1],
                         [-1, 1],
                         [-1. / 3., -1],
                         [1. / 3., -1],
                         [1. / 3., 1],
                         [-1. / 3., 1]])
    vtk_cells = [[0, 4], [4, 5], [5, 1], [1, 2], [
        2, 6], [6, 7], [7, 3], [3, 0], [4, 7], [5, 6]]
    vtk_cell_types = 'Line'

    def _get_ip_coords(self):
        return array([[-1., 0., 0.], [-1. / 3., 0., 0.], [1. / 3., 0., 0.], [1., 0., 0.]], dtype=float_)

    def _get_ip_weights(self):
        return array([[3. / 8.], [9. / 8.], [9. / 8.], [3. / 8.]], dtype=float_)

    def get_N_geo_mtx(self, r_pnt):
        '''
        Return geometric shape functions
        @param r_pnt:
        '''
        cx = array(self.geo_r, dtype=float_)
        Nr = array([[1 / 4. * (1 + r_pnt[0] * cx[i, 0]) * (1 + r_pnt[1] * cx[i, 1])
                     for i in range(0, 4)]], dtype=float_)
        return Nr

    def get_N_mtx(self, r_pnt):
        '''
        Return shape functions
        @param r_pnt:local coordinates
        '''
        # generate the shape functions and use r_pnt[1] - y-direction to distinguish
        # the phase of the material
        #
        # r_pnt[1] < 0 corresponds to phase1
        # r_pnt[1] > 0 corresponds to phase2
        #
        # switching is realized using the function
        # _get_one_if_same_sign - if the r_pnt[1] has the same sign
        # as the nodal coordinate geo_r[i,1] then 1.0 is returned,
        # 0.0 otherwise
        #
        r = r_pnt[0]
        s = r_pnt[1]
        N_bar_mtx = array([-1. / 16. * (r - 1.) * (9 * r ** 2 - 1.),
                           9. / 16. * (3. * r - 1.) * (r ** 2 - 1.),
                           - 9. / 16. * (3. * r + 1.) * (r ** 2 - 1.),
                           1. / 16. * (r + 1.) * (9 * r ** 2 - 1.)], dtype=float_)
        # blow the matrix for the DOFs using the helper matrices above
        #
        N_mtx = zeros((2, 16), dtype=float_)

        N_mtx[0, [0, 8, 10, 2]] = N_bar_mtx * _get_one_if_same_sign(s, -1.)
        N_mtx[0, [6, 14, 12, 4]] = N_bar_mtx * _get_one_if_same_sign(s, 1.)
        N_mtx[1, [1, 9, 11, 3]] = N_bar_mtx * _get_one_if_same_sign(s, -1.)
        N_mtx[1, [7, 15, 13, 5]] = N_bar_mtx * _get_one_if_same_sign(s, 1.)
        return N_mtx

    def get_B_mtx(self, r, X):
        '''
        Return kinematic matrix
        @param r:local coordinates
        @param X:global coordinates
        '''
        # length in x-direction
        L = (X[1, 0] - X[0, 0])

        # generate the shape functions of the form
        # N[0], N[1], N[2], N[3]
        #
        N = array([-1. / 16. * (r[0] - 1.) * (9 * r[0] ** 2 - 1.),
                   9. / 16. * (3. * r[0] - 1.) * (r[0] ** 2 - 1.),
                   - 9. / 16. * (3. * r[0] + 1.) * (r[0] ** 2 - 1.),
                   1. / 16. * (r[0] + 1.) * (9 * r[0] ** 2 - 1.)], dtype=float_)

        dN = array([-27. / 16. * r[0] ** 2 + 1. / 16. + 9. / 8. * r[0],
                    81. / 16. * r[0] ** 2 - 27. / 16. - 9. / 8. * r[0],
                    - 81. / 16. * r[0] ** 2 + 27. / 16. - 9. / 8. * r[0],
                    27. / 16. * r[0] ** 2 - 1. / 16. + 9. / 8. * r[0]], dtype=float_) * 2. / L

        # assemble the B matrix mapping the DOFs to strains and slip and opening
        #                  u0,   v0,   u1,  v1,  u2,    v2,   u3,   v3,  u4,    v4,   u5,   v5,   u6,  v6,   u7,   v7
        B_mtx = array([[dN[0], 0, dN[3], 0, 0, 0, 0, 0, dN[1], 0, dN[2], 0, 0, 0, 0, 0],  # eps_1
                       [N[0], 0, N[3], 0, -N[3], 0, -N[0], 0, N[1],
                           0, N[2], 0, -N[2], 0, -N[1], 0],  # slip
                       [0, N[0], 0, N[3], 0, -N[3], 0, -N[0], 0, N[1],
                           0, N[2], 0, -N[2], 0, -N[1]],  # opening
                       [0, 0, 0, 0, dN[3], 0, dN[0], 0, 0, 0,
                           0, 0, dN[2], 0, dN[1], 0]  # eps_2
                       ], dtype=float_)
        return B_mtx

    def get_eps1(self, sctx, u):
        '''Get strain in phase 1.
        '''
        eps = self.get_eps_eng(sctx, u)
        eps1 = eps[[0]]
        return eps1

    def get_eps2(self, sctx, u):
        '''Get strain in phase 2.
        '''
        eps = self.get_eps_eng(sctx, u)
        eps2 = eps[[3]]
        return eps2

    def get_slip(self, sctx, u):
        '''Get slip and opening.
        '''
        eps = self.get_eps_eng(sctx, u)
        slip = eps[1:3]
        return slip

    def _rte_dict_default(self):
        rte_dict = super(FETS1D52L8ULRH, self)._rte_dict_default()
        del rte_dict['eps_app']  # the epsilon does not have a form of a tensor
        rte_dict['slip'] = RTraceEvalElemFieldVar(eval=self.get_slip, ts=self)
        rte_dict['eps1'] = RTraceEvalElemFieldVar(eval=self.get_eps1, ts=self)
        rte_dict['eps2'] = RTraceEvalElemFieldVar(eval=self.get_eps2, ts=self)
        return rte_dict

    def _get_J_det(self, r_pnt3d, X_mtx):
        return (X_mtx[1, 0] - X_mtx[0, 0]) / 2.


def _get_one_if_same_sign(a, b):
    '''Helper function returning 1 if sign(a) == sign(b)
    and zero otherwise.
    '''
    sa = fabs(a) / a
    sb = fabs(b) / b
    return fabs(1. / 2. * (sa + sb))

#----------------------- example --------------------


def example():
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine
    from ibvpy.mesh.fe_grid import FEGrid

    A_fiber = 1.
    E_fiber = 1.
    stiffness_fiber = E_fiber * A_fiber

    d = 2 * sqrt(Pi)
    tau_max = 0.1 * d * Pi
    G = 100
    f_max = 0.2
    mats_eval = MATS1D5Bond(mats_phase1=MATS1DElastic(E=stiffness_fiber),
                            mats_phase2=MATS1DElastic(E=0),
                            mats_ifslip=MATS1DPlastic(E=G,
                                                      sigma_y=tau_max,
                                                      K_bar=0.,
                                                      H_bar=0.),
                            mats_ifopen=MATS1DElastic(E=0))

    fets_eval = FETS1D52L8ULRH(mats_eval=mats_eval)
    domain = FEGrid(coord_max=(1., 0.2),
                    shape=(16, 1),
                    fets_eval=fets_eval)

    end_dof = domain[-1, 0, -1, 0].dofs[0, 0, 0]
    ts = TS(dof_resultants=True,
            sdomain=domain,
            # conversion to list (square brackets) is only necessary for slicing of
            # single dofs, e.g "get_left_dofs()[0,1]"
            bcond_list=[
                BCSlice(var='u', value=0., dims=[0],
                        slice=domain[:, :, :, -1]),
                BCSlice(var='u', value=0., dims=[1],
                        slice=domain[:, :, :, :]),
                BCSlice(var='f', value=f_max, dims=[0],
                        slice=domain[-1, 0, -1, 0])
            ],
            rtrace_list=[RTDofGraph(name='Fi,right over u_right (iteration)',
                                    var_y='F_int', idx_y=end_dof,
                                    var_x='U_k', idx_x=end_dof),
                         RTraceDomainListField(name='slip',
                                               var='slip', idx=0),
                         RTraceDomainListField(name='eps1',
                                               var='eps1', idx=0),
                         RTraceDomainListField(name='eps2',
                                               var='eps2', idx=0),
                         RTraceDomainListField(name='shear_flow',
                                               var='shear_flow', idx=0),
                         RTraceDomainListField(name='sig1',
                                               var='sig1', idx=0),
                         RTraceDomainListField(name='sig2',
                                               var='sig2', idx=0),
                         RTraceDomainListField(name='Displacement',
                                               var='u', idx=0)
                         ])

    # Add the time-loop control
    tloop = TLoop(tstepper=ts, KMAX=30, debug=False,
                  tline=TLine(min=0.0, step=0.1, max=1.0))

    print(tloop.eval())
    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tloop)
    app.main()


if __name__ == '__main__':
    example()
