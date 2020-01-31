
from numpy import \
    array, zeros, dot, hstack, \
    identity
from scipy.linalg import \
    inv
from traits.api import \
    Array, Float, \
    Instance, Int

from ibvpy.fets.fets_eval import FETSEval
from ibvpy.mats.mats_eval import MATSEval


#-------------------------------------------------------------------------
# FETS2D4Q9U - 9 nodes subparametric quadrilateral (2D, quadratic, Lagrange familiy)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Element Information:
#-------------------------------------------------------------------------
#
# The order of the field approximation is higher then the order of the geometry
# approximation (subparametric element).
# The implemented shape functions are derived (in femple) based
# on the following ordering of the nodes of the parent element.
#
#    _node_coord_map_dof = Array( Float, (9,2),
#                                 [[ -1.,-1. ],
#                                  [  1.,-1. ],
#                                  [  1., 1. ],
#                                  [ -1., 1. ],
#                                  [  0.,-1. ],
#                                  [  1., 0. ],
#                                  [  0., 1. ],
#                                  [ -1., 0. ],
#                                  [  0., 0. ]])
#
# The ordering of the nodes of the parent element used for the geometry approximation
# is defined in '_node_coord_map_geo' (see code below)
# and the (linear) shape functions are derived by formula
#
#-------------------------------------------------------------------------
#
class FETS2D4Q9U(FETSEval):
    debug_on = True

    mats_eval = Instance(MATSEval)

    # Dimensional mapping
    dim_slice = slice(0, 2)

    n_e_dofs = Int(2 * 9)
    t = Float(1.0, label='thickness')

    # Integration parameters
    #
    ngp_r = 3
    ngp_s = 3

    dof_r = Array(value=[[-1., -1.],
                         [1., -1.],
                         [1., 1.],
                         [-1., 1.],
                         [0., -1.],
                         [1., 0.],
                         [0., 1.],
                         [-1., 0.],
                         [0., 0.]])
    geo_r = Array(value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])
    #
    vtk_cell_types = 'QuadraticQuad'
    vtk_r = Array(value=[[-1., -1.],
                         [0., -1.],
                         [1., -1.],
                         [-1., 0.],
                         [0., 0.],
                         [1., 0.],
                         [-1., 1.],
                         [0., 1.],
                         [1., 1.]])
    vtk_cells = [[0, 2, 8, 6, 1, 5, 7, 3, 4]]

    n_nodal_dofs = 2

    # Ordering of the nodes of the parent element used for the geometry
    # approximation
    _node_coord_map_geo = Array(Float, (4, 2),
                                [[-1., -1.],
                                 [1., -1.],
                                 [1., 1.],
                                 [-1., 1.]])

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------
    def get_N_geo_mtx(self, r_pnt):
        '''
        Return the value of shape functions for the specified local coordinate r
        '''
        cx = self._node_coord_map_geo
        N_geo_mtx = array(
            [[1 / 4. * (1 + r_pnt[0] * cx[i, 0]) * (1 + r_pnt[1] * cx[i, 1]) for i in range(0, 4)]])
        return N_geo_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives.
        Used for the conrcution of the Jacobi matrix.

        @TODO - the B matrix is used
        just for uniaxial bar here with a trivial differential
        operator.
        '''
        cx = self._node_coord_map_geo
        dNr_geo_mtx = array([[1 / 4. * cx[i, 0] * (1 + r_pnt[1] * cx[i, 1]) for i in range(0, 4)],
                             [1 / 4. * cx[i, 1] * (1 + r_pnt[0] * cx[i, 0]) for i in range(0, 4)]])
        return dNr_geo_mtx

    #---------------------------------------------------------------------
    # Method delivering the shape functions for the field variables and their derivatives
    #---------------------------------------------------------------------
    def get_N_mtx(self, r_pnt):
        '''
        Returns the matrix of the shape functions used for the field approximation
        containing zero entries. The number of rows corresponds to the number of nodal
        dofs. The matrix is evaluated for the specified local coordinate r.
        '''
        N_dof = zeros((1, 9), dtype='float_')
        N_dof[0, 0] = (
            r_pnt[0] * r_pnt[1] * (-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 0.4e1
        N_dof[0, 1] = (
            r_pnt[0] * r_pnt[1] * (-1 + r_pnt[1]) * (1 + r_pnt[0])) / 0.4e1
        N_dof[0, 2] = (
            r_pnt[0] * r_pnt[1] * (1 + r_pnt[1]) * (1 + r_pnt[0])) / 0.4e1
        N_dof[0, 3] = (
            r_pnt[0] * r_pnt[1] * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 0.4e1
        N_dof[0, 4] = -  (r_pnt[1] * (-1 + r_pnt[0])
                          * (1 + r_pnt[0]) * (-1 + r_pnt[1])) / 0.2e1
        N_dof[0, 5] = -  (r_pnt[0] * (-1 + r_pnt[1])
                          * (1 + r_pnt[1]) * (1 + r_pnt[0])) / 0.2e1
        N_dof[0, 6] = -  (r_pnt[1] * (-1 + r_pnt[0])
                          * (1 + r_pnt[0]) * (1 + r_pnt[1])) / 0.2e1
        N_dof[0, 7] = -  (r_pnt[0] * (-1 + r_pnt[1])
                          * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 0.2e1
        N_dof[0, 8] = (-1 + r_pnt[1]) * (1 + r_pnt[1]) * \
            (-1 + r_pnt[0]) * (1 + r_pnt[0])
        I_mtx = identity(self.n_nodal_dofs, float)
        N_mtx_list = [I_mtx * N_dof[0, i] for i in range(0, N_dof.shape[1])]
        N_mtx = hstack(N_mtx_list)
        return N_mtx

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions used for the field approximation
        '''
        dNr_mtx = zeros((2, 9), dtype='float_')
        dNr_mtx[0, 0] = (r_pnt[1] * (-1 + r_pnt[1]) * (-1 + r_pnt[0])
                         ) / 4.0 + (r_pnt[0] * r_pnt[1] * (-1 + r_pnt[1])) / 4.0
        dNr_mtx[0, 1] = (r_pnt[1] * (-1 + r_pnt[1]) * (1 + r_pnt[0])
                         ) / 4.0 + (r_pnt[0] * r_pnt[1] * (-1 + r_pnt[1])) / 4.0
        dNr_mtx[0, 2] = (r_pnt[1] * (1 + r_pnt[1]) * (1 + r_pnt[0])
                         ) / 4.0 + (r_pnt[0] * r_pnt[1] * (1 + r_pnt[1])) / 4.0
        dNr_mtx[0, 3] = (r_pnt[1] * (1 + r_pnt[1]) * (-1 + r_pnt[0])
                         ) / 4.0 + (r_pnt[0] * r_pnt[1] * (1 + r_pnt[1])) / 4.0
        dNr_mtx[0, 4] = - (r_pnt[1] * (-1 + r_pnt[1]) * (1 + r_pnt[0])) / \
            0.2e1 - (r_pnt[1] * (-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0
        dNr_mtx[0, 5] = - ((-1 + r_pnt[1]) * (1 + r_pnt[1]) * (1 + r_pnt[0])
                           ) / 2.0 - (r_pnt[0] * (-1 + r_pnt[1]) * (1 + r_pnt[1])) / 2.0
        dNr_mtx[0, 6] = - (r_pnt[1] * (1 + r_pnt[1]) * (1 + r_pnt[0])
                           ) / 2.0 - (r_pnt[1] * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0
        dNr_mtx[0, 7] = - ((-1 + r_pnt[1]) * (1 + r_pnt[1]) * (-1 + r_pnt[0])
                           ) / 2.0 - (r_pnt[0] * (-1 + r_pnt[1]) * (1 + r_pnt[1])) / 2.0
        dNr_mtx[0, 8] = (-1 + r_pnt[1]) * (1 + r_pnt[1]) * (1 +
                                                            r_pnt[0]) + (-1 + r_pnt[1]) * (1 + r_pnt[1]) * (-1 + r_pnt[0])
        dNr_mtx[1, 0] = (r_pnt[0] * (-1 + r_pnt[1]) * (-1 + r_pnt[0])
                         ) / 4.0 + (r_pnt[0] * r_pnt[1] * (-1 + r_pnt[0])) / 4.0
        dNr_mtx[1, 1] = (r_pnt[0] * (-1 + r_pnt[1]) * (1 + r_pnt[0])
                         ) / 4.0 + (r_pnt[0] * r_pnt[1] * (1 + r_pnt[0])) / 4.0
        dNr_mtx[1, 2] = (r_pnt[0] * (1 + r_pnt[1]) * (1 + r_pnt[0])
                         ) / 4.0 + (r_pnt[0] * r_pnt[1] * (1 + r_pnt[0])) / 4.0
        dNr_mtx[1, 3] = (r_pnt[0] * (1 + r_pnt[1]) * (-1 + r_pnt[0])
                         ) / 4.0 + (r_pnt[0] * r_pnt[1] * (-1 + r_pnt[0])) / 4.0
        dNr_mtx[1, 4] = - ((-1 + r_pnt[0]) * (1 + r_pnt[0]) * (-1 + r_pnt[1])
                           ) / 2.0 - (r_pnt[1] * (-1 + r_pnt[0]) * (1 + r_pnt[0])) / 2.0
        dNr_mtx[1, 5] = - (r_pnt[0] * (1 + r_pnt[1]) * (1 + r_pnt[0])
                           ) / 2.0 - (r_pnt[0] * (-1 + r_pnt[1]) * (1 + r_pnt[0])) / 2.0
        dNr_mtx[1, 6] = - ((-1 + r_pnt[0]) * (1 + r_pnt[0]) * (1 + r_pnt[1])
                           ) / 2.0 - (r_pnt[1] * (-1 + r_pnt[0]) * (1 + r_pnt[0])) / 2.0
        dNr_mtx[1, 7] = - (r_pnt[0] * (1 + r_pnt[1]) * (-1 + r_pnt[0])
                           ) / 2.0 - (r_pnt[0] * (-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0
        dNr_mtx[1, 8] = (-1 + r_pnt[0]) * (1 + r_pnt[0]) * (1 +
                                                            r_pnt[1]) + (-1 + r_pnt[0]) * (1 + r_pnt[0]) * (-1 + r_pnt[1])
        return dNr_mtx

    def get_B_mtx(self, r_pnt, X_mtx):
        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_mtx = self.get_dNr_mtx(r_pnt)
        dNx_mtx = dot(inv(J_mtx), dNr_mtx)
        Bx_mtx = zeros((3, 18), dtype='float_')
        for i in range(0, 9):
            Bx_mtx[0, i * 2] = dNx_mtx[0, i]
            Bx_mtx[1, i * 2 + 1] = dNx_mtx[1, i]
            Bx_mtx[2, i * 2] = dNx_mtx[1, i]
            Bx_mtx[2, i * 2 + 1] = dNx_mtx[0, i]
        return Bx_mtx

#----------------------- example with the new domain --------
if __name__ == '__main__':
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine, BCDofGroup

    #from lib.mats.mats2D.mats_cmdm2D.mats_mdm2d import MACMDM
    from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
    #from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
    from ibvpy.mesh.fe_grid import FEGrid

    fets_eval = FETS2D4Q9U(mats_eval=MATS2DScalarDamage())
    #fets_eval = FETS2D4Q9U(mats_eval=MATS2DElastic())

    # Discretization
    #
    domain = FEGrid(coord_max=(3., 3., 0.),
                    shape=(3, 3),
                    fets_eval=fets_eval)
    print('n_dofs', domain.n_dofs)

    right_dof = 2
    ts = TS(
        sdomain=domain,
        # conversion to list (square brackets) is only necessary for slicing of
        # single dofs, e.g "get_left_dofs()[0,1]"

        bcond_list=[BCDofGroup(var='u', value=0., dims=[0],
                               get_dof_method=domain.get_left_dofs),
                    BCDofGroup(var='u', value=0., dims=[1],
                               get_dof_method=domain.get_bottom_left_dofs),
                    BCDofGroup(var='u', value=0.002, dims=[0],
                               get_dof_method=domain.get_right_dofs)],

        rtrace_list=[RTDofGraph(name='Fi,right over u_right (iteration)',
                                 var_y='F_int', idx_y=right_dof,
                                 var_x='U_k', idx_x=right_dof),
                     #                         RTraceDomainListField(name = 'Stress' ,
                     #                         var = 'sig_app', idx = 0,
                     #                         record_on = 'update'),
                     RTraceDomainListField(name='Displacement',
                                                var='u', idx=0),
                     #                             RTraceDomainListField(name = 'N0' ,
                     #                                          var = 'N_mtx', idx = 0,
                     # record_on = 'update')

                     ]
    )

    # Add the time-loop control

    global tloop
    tloop = TLoop(tstepper=ts,
                  DT=0.5,
                  tline=TLine(min=0.0,  max=1.0, step=0.1))

    import cProfile
    cProfile.run('tloop.eval()', 'tloop_prof')

    import pstats
    p = pstats.Stats('tloop_prof')
    p.strip_dirs()
    print('cumulative')
    p.sort_stats('cumulative').print_stats(20)
    print('time')
    p.sort_stats('time').print_stats(20)

    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tloop)
    app.main()
