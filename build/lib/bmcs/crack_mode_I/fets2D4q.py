
import sys

from ibvpy.dots.dots_grid_eval import DOTSGridEval
from ibvpy.fets.fets_eval import FETSEval
from scipy.linalg import \
    inv
from traits.api import \
    Array, Float, Int, Property, cached_property, \
    List

import numpy as np
import sympy as sp


r_, s_ = sp.symbols('r, s')


print(sys.path)


#-------------------------------------------------------------------------
# FETS2D4Q - 4 nodes iso-parametric quadrilateral element (2D, linear, Lagrange family)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Element Information:
#-------------------------------------------------------------------------
#
# Here an isoparametric element formulation is applied.
# The implemented shape functions are derived based on the
# ordering of the nodes of the parent element defined in
# '_node_coord_map' (see below)
#
#-------------------------------------------------------------------------


class FETS2D4Q(FETSEval):

    debug_on = True

    dots_class = DOTSGridEval

    # Dimensional mapping
    dim_slice = slice(0, 2)

    # Order of node positions for the formulation of shape function
    #
    dof_r = Array(value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])
    geo_r = Array(value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])

    n_e_dofs = Int(8,
                   FE=True,
                   enter_set=False, auto_set=True,
                   label='number of elemnet dofs'
                   )
    t = Float(1.0,
              CS=True,
              enter_set=False, auto_set=True,
              label='thickness')

    # Integration parameters
    #
    ngp_r = Int(2,
                FE=True,
                enter_set=False, auto_set=True,
                )
    ngp_s = Int(2, FE=True,
                enter_set=False, auto_set=True,
                )

    # Corner nodes are used for visualization
    vtk_r = Array(value=[[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    vtk_cells = [[0, 1, 2, 3]]
    vtk_cell_types = 'Quad'

    #vtk_point_ip_map = [0,1,3,2]
    n_nodal_dofs = Int(2)

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------
    def get_N_geo_mtx(self, r_pnt):
        '''
        Return the value of shape functions for the specified local coordinate r
        '''
        cx = np.array(self.geo_r, dtype='float_')
        Nr = np.array([[1 / 4. * (1 + r_pnt[0] * cx[i, 0]) * (1 + r_pnt[1] * cx[i, 1])
                        for i in range(0, 4)]])
        return Nr

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives.
        Used for the conrcution of the Jacobi matrix.

        @TODO - the B matrix is used
        just for uniaxial bar here with a trivial differential
        operator.
        '''
        #cx = self._node_coord_map
        cx = np.array(self.geo_r, dtype='float_')
        dNr_geo = np.array([[1 / 4. * cx[i, 0] * (1 + r_pnt[1] * cx[i, 1])
                             for i in range(0, 4)],
                            [1 / 4. * cx[i, 1] * (1 + r_pnt[0] * cx[i, 0])
                             for i in range(0, 4)]])
        return dNr_geo

    #---------------------------------------------------------------------
    # Method delivering the shape functions for the field variables and their derivatives
    #---------------------------------------------------------------------
    def get_N_mtx(self, r_pnt):
        '''
        Returns the matrix of the shape functions used for the field approximation
        containing zero entries. The number of rows corresponds to the number of nodal
        dofs. The matrix is evaluated for the specified local coordinate r.
        '''
        Nr_geo = self.get_N_geo_mtx(r_pnt)
        I_mtx = np.identity(self.n_nodal_dofs, float)
        N_mtx_list = [I_mtx * Nr_geo[0, i] for i in range(0, Nr_geo.shape[1])]
        N_mtx = np.hstack(N_mtx_list)
        return N_mtx

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions
        '''
        return self.get_dNr_geo_mtx(r_pnt)

    def get_B_mtx(self, r_pnt, X_mtx):
        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_mtx = self.get_dNr_mtx(r_pnt)
        dNx_mtx = np.dot(inv(J_mtx), dNr_mtx)
        Bx_mtx = np.zeros((3, 8), dtype='float_')
        for i in range(0, 4):
            Bx_mtx[0, i * 2] = dNx_mtx[0, i]
            Bx_mtx[1, i * 2 + 1] = dNx_mtx[1, i]
            Bx_mtx[2, i * 2] = dNx_mtx[1, i]
            Bx_mtx[2, i * 2 + 1] = dNx_mtx[0, i]
        return Bx_mtx

    r_m = Array(value=[[-1], [1]], dtype=np.float_)
    w_m = Array(value=[1, 1], dtype=np.float_)

    dNr_i_geo = List([- 1.0 / 2.0,
                      1.0 / 2.0, ])

    Nr_i = Nr_i_geo
    dNr_i = dNr_i_geo

    N_mi_geo = Property()

    @cached_property
    def _get_N_mi_geo(self):
        self.get_N_mi(sp.Matrix(self.Nr_i_geo, dtype=np.float_))

    dN_mid_geo = Property()

    @cached_property
    def _get_dN_mid_geo(self):
        return self.get_dN_mid(sp.Matrix(self.dNr_i_geo, dtype=np.float_))

    N_mi = Property()

    @cached_property
    def _get_N_mi(self):
        return self.get_N_mi(sp.Matrix(self.Nr_i, dtype=np.float_))

    dN_mid = Property()

    @cached_property
    def _get_dN_mid(self):
        return self.get_dN_mid(sp.Matrix(self.dNr_i, dtype=np.float_))

    def get_N_mi(self, Nr_i):
        return np.array([Nr_i.subs(r_, r)
                         for r in self.r_m], dtype=np.float_)

    def get_dN_mid(self, dNr_i):
        dN_mdi = np.array([[dNr_i.subs(r_, r)]
                           for r in self.r_m], dtype=np.float_)
        return np.einsum('mdi->mid', dN_mdi)

    n_m = Property(depends_on='w_m')

    @cached_property
    def _get_n_m(self):
        return len(self.w_m)

    A_C = Property(depends_on='A_m,A_f')

    @cached_property
    def _get_A_C(self):
        return np.array((self.A_f, self.P_b, self.A_m), dtype=np.float_)

    def get_B_EmisC(self, J_inv_Emdd):
        fets_eval = self

        n_dof_r = fets_eval.n_dof_r
        n_nodal_dofs = fets_eval.n_nodal_dofs

        n_m = fets_eval.n_gp
        n_E = J_inv_Emdd.shape[0]
        n_s = 3
        #[ d, i]
        r_ip = fets_eval.ip_coords[:, :-2].T
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)

        # shape function for the unknowns
        # [ d, n, i]
        Nr = 0.5 * (1. + geo_r[:, :, None] * r_ip[None, :])
        dNr = 0.5 * geo_r[:, :, None] * np.array([1, 1])

        # [ i, n, d ]
        Nr = np.einsum('dni->ind', Nr)
        dNr = np.einsum('dni->ind', dNr)
        Nx = Nr
        # [ n_e, n_ip, n_dof_r, n_dim_dof ]
        dNx = np.einsum('eidf,inf->eind', J_inv_Emdd, dNr)

        B = np.zeros((n_E, n_m, n_dof_r, n_s, n_nodal_dofs), dtype='f')
        B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
        B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
        B_factors = np.array([-1, 1], dtype='float_')
        B[:, :, :, B_N_n_rows, B_N_n_cols] = (B_factors[None, None, :] *
                                              Nx[:, :, N_idx])
        B[:, :, :, B_dN_n_rows, B_dN_n_cols] = dNx[:, :, :, dN_idx]

        return B

    def get_B_EimsC(self, dN_Eimd, sN_Cim):

        n_E, n_i, n_m, n_d = dN_Eimd.shape
        n_C, n_i, n_m = sN_Cim.shape
        n_s = 3
        B_EimsC = np.zeros(
            (n_E, n_i, n_m, n_s, n_C), dtype='f')
        B_EimsC[..., [1, 1], [0, 1]] = sN_Cim[[0, 1], :, :]
        B_EimsC[..., [0, 2], [0, 1]] = dN_Eimd[:, [0, 1], :, :]

        return B_EimsC


#----------------------- example --------------------

def example_with_new_domain():
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, \
        RTraceDomainListInteg, TLoop, \
        TLine, BCDof, DOTSEval
    from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
    from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage

    from ibvpy.api import BCDofGroup

    mats_eval = MATS2DElastic()
    fets_eval = FETS2D4Q(mats_eval=mats_eval)
    #fets_eval = FETS2D4Q(mats_eval = MATS2DScalarDamage())

    print(fets_eval.vtk_node_cell_data)

    from ibvpy.mesh.fe_grid import FEGrid
    from ibvpy.mesh.fe_refinement_grid import FERefinementGrid
    from ibvpy.mesh.fe_domain import FEDomain
    from mathkit.mfn import MFnLineArray

    # Discretization
    fe_grid = FEGrid(coord_max=(10., 4., 0.),
                     shape=(10, 3),
                     fets_eval=fets_eval)

    bcg = BCDofGroup(var='u', value=0., dims=[0],
                     get_dof_method=fe_grid.get_left_dofs)
    bcg.setup(None)
    print('labels', bcg._get_labels())
    print('points', bcg._get_mvpoints())

    mf = MFnLineArray(  # xdata = arange(10),
        ydata=np.array([0, 1, 2, 3]))

    right_dof = 2
    tstepper = TS(sdomain=fe_grid,
                  bcond_list=[BCDofGroup(var='u', value=0., dims=[0, 1],
                                         get_dof_method=fe_grid.get_left_dofs),
                              #                                   BCDofGroup( var='u', value = 0., dims = [1],
                              # get_dof_method = fe_grid.get_bottom_dofs ),
                              BCDofGroup(var='u', value=.005, dims=[1],
                                         time_function=mf.get_value,
                                         get_dof_method=fe_grid.get_right_dofs)],
                  rtrace_list=[
                      RTDofGraph(name='Fi,right over u_right (iteration)',
                                 var_y='F_int', idx_y=right_dof,
                                 var_x='U_k', idx_x=right_dof,
                                 record_on='update'),
                      RTraceDomainListField(name='Stress',
                                            var='sig_app', idx=0,
                                            position='int_pnts',
                                            record_on='update'),
                      #                     RTraceDomainListField(name = 'Damage' ,
                      #                                    var = 'omega', idx = 0,
                      #
                      #                    record_on = 'update',
                      #                                    warp = True),
                      RTraceDomainListField(name='Displacement',
                                            var='u', idx=0,
                                            record_on='update',
                                            warp=True),
                      RTraceDomainListField(name='Strain energy',
                                            var='strain_energy', idx=0,
                                            record_on='update',
                                            warp=False),
                      RTraceDomainListInteg(name='Integ strain energy',
                                            var='strain_energy', idx=0,
                                            record_on='update',
                                            warp=False),
                      #                    RTraceDomainListField(name = 'N0' ,
                      #                                      var = 'N_mtx', idx = 0,
                      # record_on = 'update')
                  ]
                  )

    # Add the time-loop control
    #global tloop
    tloop = TLoop(tstepper=tstepper, KMAX=300, tolerance=1e-4,
                  tline=TLine(min=0.0, step=1.0, max=1.0))

    #import cProfile
    #cProfile.run('tloop.eval()', 'tloop_prof' )
    # print tloop.eval()
    #import pstats
    #p = pstats.Stats('tloop_prof')
    # p.strip_dirs()
    # print 'cumulative'
    # p.sort_stats('cumulative').print_stats(20)
    # print 'time'
    # p.sort_stats('time').print_stats(20)

    tloop.eval()
    # Put the whole thing into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tloop)
    app.main()


if __name__ == '__main__':
    example_with_new_domain()
