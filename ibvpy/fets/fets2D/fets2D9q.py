
from scipy.linalg import \
    inv
from traits.api import \
    Array, Float, \
    Instance, Int
from ibvpy.fets.fets_eval import FETSEval
from ibvpy.mats.mats_eval import MATSEval
import numpy as np


#------------------------------------------------------------------------------
# FETS2D9Q - 9 nodes isoparametric quadrilateral (2D, quadratic, Lagrange family)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Element Information:
#------------------------------------------------------------------------------
#
# Here an isoparametric element formulation is applied.
# The implemented shape functions are derived in femple
# based on the following ordering of the nodes of the
# parent element:
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
#    _node_coord_map_geo = Array( Float, (9,2),
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
#------------------------------------------------------------------------------
class FETS2D9Q(FETSEval):
    debug_on = True

    mats_eval = Instance(MATSEval)

    # Dimensional mapping
    dim_slice = slice(0, 2)

    n_e_dofs = Int(9 * 2)
    t = Float(1.0, label='thickness')
    E = Float(1.0, label="Young's modulus")
    nu = Float(0., label="Poisson's ratio")

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
    geo_r = Array(value=[[-1., -1.],
                         [1., -1.],
                         [1., 1.],
                         [-1., 1.],
                         [0., -1.],
                         [1., 0.],
                         [0., 1.],
                         [-1., 0.],
                         [0., 0.]])
    #
    vtk_cell_types = 'QuadraticQuad'
    vtk_r = Array(value=[[-1., -1.],
                         [1., -1.],
                         [1., 1.],
                         [-1., 1.],
                         [0., -1.],
                         [1., 0.],
                         [0., 1.],
                         [-1., 0.],
                         [0., 0.]])
    vtk_cells = [[0, 1, 2, 3, 4, 5, 6, 7, 8]]

    n_nodal_dofs = Int(2)

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------
    def get_N_geo_mtx(self, r_pnt):
        '''
        Return the value of shape functions for the specified local coordinate r
        '''
        N_geo_mtx = np.zeros((1, 9), dtype='float_')
        N_geo_mtx[0, 0] = (
            r_pnt[0] * r_pnt[1] * (-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 4.0
        N_geo_mtx[0, 1] = (
            r_pnt[0] * r_pnt[1] * (-1 + r_pnt[1]) * (1 + r_pnt[0])) / 4.0
        N_geo_mtx[0, 2] = (
            r_pnt[0] * r_pnt[1] * (1 + r_pnt[1]) * (1 + r_pnt[0])) / 4.0
        N_geo_mtx[0, 3] = (
            r_pnt[0] * r_pnt[1] * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 4.0
        N_geo_mtx[
            0, 4] = - (r_pnt[1] * (-1 + r_pnt[0]) * (1 + r_pnt[0]) * (-1 + r_pnt[1])) / 2.0
        N_geo_mtx[
            0, 5] = - (r_pnt[0] * (-1 + r_pnt[1]) * (1 + r_pnt[1]) * (1 + r_pnt[0])) / 2.0
        N_geo_mtx[
            0, 6] = - (r_pnt[1] * (-1 + r_pnt[0]) * (1 + r_pnt[0]) * (1 + r_pnt[1])) / 2.0
        N_geo_mtx[
            0, 7] = - (r_pnt[0] * (-1 + r_pnt[1]) * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0
        N_geo_mtx[0, 8] = (-1 + r_pnt[1]) * \
            (1 + r_pnt[1]) * (-1 + r_pnt[0]) * (1 + r_pnt[0])
        return N_geo_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives.
        Used for the construction of the Jacobi matrix.
        '''
        dNr_geo_mtx = np.zeros((2, 9), dtype='float_')
        dNr_geo_mtx[0, 0] = (r_pnt[1] * (-1 + r_pnt[1]) * (-1 + r_pnt[0])
                             ) / 4.0 + (r_pnt[0] * r_pnt[1] * (-1 + r_pnt[1])) / 4.0
        dNr_geo_mtx[0, 1] = (r_pnt[1] * (-1 + r_pnt[1]) * (1 + r_pnt[0])
                             ) / 4.0 + (r_pnt[0] * r_pnt[1] * (-1 + r_pnt[1])) / 4.0
        dNr_geo_mtx[0, 2] = (r_pnt[1] * (1 + r_pnt[1]) * (1 + r_pnt[0])
                             ) / 4.0 + (r_pnt[0] * r_pnt[1] * (1 + r_pnt[1])) / 4.0
        dNr_geo_mtx[0, 3] = (r_pnt[1] * (1 + r_pnt[1]) * (-1 + r_pnt[0])
                             ) / 4.0 + (r_pnt[0] * r_pnt[1] * (1 + r_pnt[1])) / 4.0
        dNr_geo_mtx[0, 4] = - (r_pnt[1] * (-1 + r_pnt[1]) * (1 + r_pnt[0])) / 0.2e1 - (
            r_pnt[1] * (-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0
        dNr_geo_mtx[0, 5] = - ((-1 + r_pnt[1]) * (1 + r_pnt[1]) * (1 + r_pnt[0])) / 2.0 - (
            r_pnt[0] * (-1 + r_pnt[1]) * (1 + r_pnt[1])) / 2.0
        dNr_geo_mtx[0, 6] = - (r_pnt[1] * (1 + r_pnt[1]) * (1 + r_pnt[0])
                               ) / 2.0 - (r_pnt[1] * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0
        dNr_geo_mtx[0, 7] = - ((-1 + r_pnt[1]) * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0 - (
            r_pnt[0] * (-1 + r_pnt[1]) * (1 + r_pnt[1])) / 2.0
        dNr_geo_mtx[0, 8] = (-1 + r_pnt[1]) * (1 + r_pnt[1]) * (1 +
                                                                r_pnt[0]) + (-1 + r_pnt[1]) * (1 + r_pnt[1]) * (-1 + r_pnt[0])
        dNr_geo_mtx[1, 0] = (r_pnt[0] * (-1 + r_pnt[1]) * (-1 + r_pnt[0])
                             ) / 4.0 + (r_pnt[0] * r_pnt[1] * (-1 + r_pnt[0])) / 4.0
        dNr_geo_mtx[1, 1] = (r_pnt[0] * (-1 + r_pnt[1]) * (1 + r_pnt[0])
                             ) / 4.0 + (r_pnt[0] * r_pnt[1] * (1 + r_pnt[0])) / 4.0
        dNr_geo_mtx[1, 2] = (r_pnt[0] * (1 + r_pnt[1]) * (1 + r_pnt[0])
                             ) / 4.0 + (r_pnt[0] * r_pnt[1] * (1 + r_pnt[0])) / 4.0
        dNr_geo_mtx[1, 3] = (r_pnt[0] * (1 + r_pnt[1]) * (-1 + r_pnt[0])
                             ) / 4.0 + (r_pnt[0] * r_pnt[1] * (-1 + r_pnt[0])) / 4.0
        dNr_geo_mtx[1, 4] = - ((-1 + r_pnt[0]) * (1 + r_pnt[0]) * (-1 + r_pnt[1])) / 2.0 - (
            r_pnt[1] * (-1 + r_pnt[0]) * (1 + r_pnt[0])) / 2.0
        dNr_geo_mtx[1, 5] = - (r_pnt[0] * (1 + r_pnt[1]) * (1 + r_pnt[0])
                               ) / 2.0 - (r_pnt[0] * (-1 + r_pnt[1]) * (1 + r_pnt[0])) / 2.0
        dNr_geo_mtx[1, 6] = - ((-1 + r_pnt[0]) * (1 + r_pnt[0]) * (1 + r_pnt[1])) / 2.0 - (
            r_pnt[1] * (-1 + r_pnt[0]) * (1 + r_pnt[0])) / 2.0
        dNr_geo_mtx[1, 7] = - (r_pnt[0] * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0 - (
            r_pnt[0] * (-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0
        dNr_geo_mtx[1, 8] = (-1 + r_pnt[0]) * (1 + r_pnt[0]) * (1 +
                                                                r_pnt[1]) + (-1 + r_pnt[0]) * (1 + r_pnt[0]) * (-1 + r_pnt[1])
        return dNr_geo_mtx

    #---------------------------------------------------------------------
    # Method delivering the shape functions for the field variables and their derivatives
    #---------------------------------------------------------------------
    def get_N_mtx(self, r_pnt):
        '''
        Returns the matrix of the shape functions used for the field approximation
        containing zero entries. The number of rows corresponds to the number of nodal
        dofs. The matrix is evaluated for the specified local coordinate r_pnt.
        '''
        N = self.get_N_geo_mtx(r_pnt)
        I_mtx = np.identity(self.n_nodal_dofs, float)
        N_mtx_list = [I_mtx * N[0, i] for i in range(0, N.shape[1])]
        N_mtx = np.hstack(N_mtx_list)
        return N_mtx

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions used for the field approximation
        '''
        dNr_mtx = self.get_dNr_geo_mtx(r_pnt)
        return dNr_mtx

    def get_B_mtx(self, r_pnt, X_mtx):
        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_mtx = self.get_dNr_mtx(r_pnt)
        dNx_mtx = np.dot(inv(J_mtx), dNr_mtx)
        Bx_mtx = np.zeros((3, 18), dtype='float_')
        for i in range(0, 9):
            Bx_mtx[0, i * 2] = dNx_mtx[0, i]
            Bx_mtx[1, i * 2 + 1] = dNx_mtx[1, i]
            Bx_mtx[2, i * 2] = dNx_mtx[1, i]
            Bx_mtx[2, i * 2 + 1] = dNx_mtx[0, i]
        return Bx_mtx

#----------------------- example --------------------

if __name__ == '__main__':
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine, BCDofGroup

    #from lib.mats.mats2D.mats_cmdm2D.mats_mdm2d import MACMDM
    #from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
    from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic

    fets_eval = FETS2D9Q(mats_eval=MATS2DElastic())
    #fets_eval = FETS2D9Q(mats_eval = MACMDM())

    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid(coord_max=(3., 3., 0.),
                    shape=(4, 4),
                    fets_eval=fets_eval)

    right_dof = 2
    ts = TS(sdomain=domain,
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
    #
    tl = TLoop(tstepper=ts,
               DT=0.5,
               tline=TLine(min=0.0,  max=1.0))

    tl.eval()
    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tl)
    app.main()
