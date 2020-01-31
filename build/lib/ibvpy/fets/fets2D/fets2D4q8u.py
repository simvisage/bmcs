
from scipy.linalg import \
    inv
from traits.api import \
    Array, Float, \
    Instance, Int

from ibvpy.fets.fets_eval import FETSEval
from ibvpy.mats.mats_eval import MATSEval
import numpy as np


#-------------------------------------------------------------------------
# FETS2D4Q8U - 8 nodes subparametric quadrilateral (2D, quadratic, serendipity family)
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
#            _node_coord_map_dof = np.array( Float, (8,2),
#                                 [[ -1.,-1. ],
#                                  [  1.,-1. ],
#                                  [  1., 1. ],
#                                  [ -1., 1. ],
#                                  [  0.,-1. ],
#                                  [  1., 0. ],
#                                  [  0., 1. ],
#                                  [ -1., 0. ]])
#
# The ordering of the nodes of the parent element used for the geometry approximation
# is defined in '_node_coord_map_geo' (see code below)
# and the (linear) shape functions are derived by formula
#
#-------------------------------------------------------------------------
class FETS2D4Q8U(FETSEval):
    debug_on = True

    mats_eval = Instance(MATSEval)

    # Dimensional mapping
    dim_slice = slice(0, 2)

    n_e_dofs = Int(2 * 8)
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
                         [-1., 0.]])
    geo_r = Array(value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])
    #
    vtk_r = Array(value=[[-1., -1.],
                         [0., -1.],
                         [1., -1.],
                         [-1., 0.],
                         [1., 0.],
                         [-1., 1.],
                         [0., 1.],
                         [1., 1.]])
    vtk_cells = [[0, 2, 7, 5, 1, 4, 6, 3]]
    vtk_cell_types = 'QuadraticQuad'

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
        Return the value of shape functions for the specified local coordinate r_pnt
        '''
        cx = self._node_coord_map_geo
        N_geo_mtx = np.array(
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
        dNr_geo_mtx = np.array([[1 / 4. * cx[i, 0] * (1 + r_pnt[1] * cx[i, 1]) for i in range(0, 4)],
                                [1 / 4. * cx[i, 1] * (1 + r_pnt[0] * cx[i, 0]) for i in range(0, 4)]])
        return dNr_geo_mtx

    #-------------------------------------------------------------------------
    # Method delivering the shape functions for the field variables and their derivatives
    #-------------------------------------------------------------------------
    def get_N_mtx(self, r_pnt):
        '''
        Returns the matrix of the shape functions (derived in femple) used for the field 
        approximation containing zero entries. The number of rows corresponds to the number 
        of nodal dofs. The matrix is evaluated for the specified local coordinate r_pnt.
        '''
        N_dof = np.zeros((1, 8))
        N_dof[0, 0] = -((-1 + r_pnt[1]) * (-1 + r_pnt[0])
                        * (r_pnt[0] + 1 + r_pnt[1])) / 4.0
        N_dof[0, 1] = -((-1 + r_pnt[1]) * (1 + r_pnt[0])
                        * (r_pnt[0] - 1 - r_pnt[1])) / 4.0
        N_dof[0, 2] = (
            (1 + r_pnt[1]) * (1 + r_pnt[0]) * (r_pnt[0] - 1 + r_pnt[1])) / 4.0
        N_dof[0, 3] = (
            (1 + r_pnt[1]) * (-1 + r_pnt[0]) * (r_pnt[0] + 1 - r_pnt[1])) / 4.0
        N_dof[0, 4] = (
            (-1 + r_pnt[0]) * (1 + r_pnt[0]) * (-1 + r_pnt[1])) / 2.0
        N_dof[0, 5] = - \
            ((-1 + r_pnt[1]) * (1 + r_pnt[1]) * (1 + r_pnt[0])) / 2.0
        N_dof[0, 6] = - \
            ((-1 + r_pnt[0]) * (1 + r_pnt[0]) * (1 + r_pnt[1])) / 2.0
        N_dof[0, 7] = (
            (-1 + r_pnt[1]) * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0

        I_mtx = np.identity(self.n_nodal_dofs, float)
        N_mtx_list = [I_mtx * N_dof[0, i] for i in range(0, N_dof.shape[1])]
        N_mtx = np.hstack(N_mtx_list)
        return N_mtx

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions (derived in femple) 
        used for the field approximation
        '''
        dNr_mtx = np.zeros((2, 8), dtype='float_')
        dNr_mtx[0, 0] = -((-1 + r_pnt[1]) * (r_pnt[0] + 1 + r_pnt[1])
                          ) / 4.0 - ((-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 4.0
        dNr_mtx[0, 1] = -((-1 + r_pnt[1]) * (r_pnt[0] - 1 - r_pnt[1])
                          ) / 4.0 - ((-1 + r_pnt[1]) * (1 + r_pnt[0])) / 4.0
        dNr_mtx[0, 2] = ((1 + r_pnt[1]) * (r_pnt[0] - 1 + r_pnt[1])
                         ) / 4.0 + ((1 + r_pnt[1]) * (1 + r_pnt[0])) / 4.0
        dNr_mtx[0, 3] = ((1 + r_pnt[1]) * (r_pnt[0] + 1 - r_pnt[1])
                         ) / 4.0 + ((1 + r_pnt[1]) * (-1 + r_pnt[0])) / 4.0
        dNr_mtx[0, 4] = ((-1 + r_pnt[1]) * (1 + r_pnt[0])) / \
            2.0 + ((-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0
        dNr_mtx[0, 5] = -((-1 + r_pnt[1]) * (1 + r_pnt[1])) / 2.0
        dNr_mtx[0, 6] = -((1 + r_pnt[1]) * (1 + r_pnt[0])) / \
            2.0 - ((1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0
        dNr_mtx[0, 7] = ((-1 + r_pnt[1]) * (1 + r_pnt[1])) / 2.0
        dNr_mtx[1, 0] = -((-1 + r_pnt[0]) * (r_pnt[0] + 1 + r_pnt[1])
                          ) / 4.0 - ((-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 4.0
        dNr_mtx[1, 1] = -((1 + r_pnt[0]) * (r_pnt[0] - 1 - r_pnt[1])
                          ) / 4.0 + ((-1 + r_pnt[1]) * (1 + r_pnt[0])) / 4.0
        dNr_mtx[1, 2] = ((1 + r_pnt[0]) * (r_pnt[0] - 1 + r_pnt[1])
                         ) / 4.0 + ((1 + r_pnt[1]) * (1 + r_pnt[0])) / 4.0
        dNr_mtx[1, 3] = ((-1 + r_pnt[0]) * (r_pnt[0] + 1 - r_pnt[1])
                         ) / 4.0 - ((1 + r_pnt[1]) * (-1 + r_pnt[0])) / 4.0
        dNr_mtx[1, 4] = ((-1 + r_pnt[0]) * (1 + r_pnt[0])) / 2.0
        dNr_mtx[1, 5] = -((1 + r_pnt[1]) * (1 + r_pnt[0])) / \
            2.0 - ((-1 + r_pnt[1]) * (1 + r_pnt[0])) / 2.0
        dNr_mtx[1, 6] = -((-1 + r_pnt[0]) * (1 + r_pnt[0])) / 2.0
        dNr_mtx[1, 7] = ((1 + r_pnt[1]) * (-1 + r_pnt[0])) / \
            2.0 + ((-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 2.0
        return dNr_mtx

    def get_B_mtx(self, r_pnt, X_mtx):
        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_mtx = self.get_dNr_mtx(r_pnt)
        dNx_mtx = np.dot(inv(J_mtx), dNr_mtx)
        Bx_mtx = np.zeros((3, 16), dtype='float_')
        for i in range(0, 8):
            Bx_mtx[0, i * 2] = dNx_mtx[0, i]
            Bx_mtx[1, i * 2 + 1] = dNx_mtx[1, i]
            Bx_mtx[2, i * 2] = dNx_mtx[1, i]
            Bx_mtx[2, i * 2 + 1] = dNx_mtx[0, i]
        return Bx_mtx


#----------------------- example --------------------

if __name__ == '__main__':
    from ibvpy.api import \
        TStepper as TS, RTraceDomainListField, TLoop, \
        TLine, BCDofGroup

    #from ibvpy.mats.mats2D.mats_cmdm2D.mats_cmdm2D import MATS2DMicroplaneDamage
    #from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
    #from ibvpy.mats.mats2D.mats2D_sdamage.strain_norm2d import *
    from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
#    fets_eval = FETS2D4Q8U(mats_eval = MACMDM())
    #fets_eval = FETS2D4Q8U(mats_eval = MATS2DScalarDamage(strain_norm_type = 'Euclidean'))
    fets_eval = FETS2D4Q8U(mats_eval=MATS2DElastic())

    # Define a mesh domain adaptor as a cached property to
    # be constracted on demand

    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid(coord_max=(3., 3., 0.),
                    shape=(1, 1),
                    fets_eval=fets_eval)

    # Put the tseval (time-stepper) into the spatial context of the
    # discretization and specify the response tracers to evaluate there.
    right_dof = 2
    ts = TS(
        sdomain=domain,
        # conversion to list (square brackets) is only necessary for slicing of
        # single dofs, e.g "get_left_dofs()[0,1]"

        # Boundary conditions for three-point-bendig:
        #         bcond_list =  [ BCDof(var='u', dof = i, value = 0.) for i in [domain.get_bottom_left_dofs()[0,0]]  ] +
        #                    [ BCDof(var='u', dof = i, value = 0.) for i in [domain.get_bottom_left_dofs()[0,1]] ] +
        #                    [ BCDof(var='u', dof = i, value = 0.) for i in [domain.get_bottom_right_dofs()[0,1]] ] +
        #                    [ BCDof(var='u', dof = i, value = 0.002 ) for i in [domain.get_top_middle_dofs()[0,1]] ],

        # Boundary conditions for shear force applied at the right border:
        bcond_list=[BCDofGroup(var='u', value=0., dims=[0, 1],
                               get_dof_method=domain.get_left_dofs),
                    #                         BCDofGroup( var='u', value = 0., dims = [1],
                    # get_dof_method = domain.get_bottom_right_dofs ),
                    BCDofGroup(var='u', value=0.002, dims=[1],
                               get_dof_method=domain.get_right_dofs)],


        rtrace_list=[
            #                         RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
            #                               var_y = 'F_int', idx_y = right_dof,
            #                               var_x = 'U_k', idx_x = right_dof),
            #                         RTraceDomainListField(name = 'Stress' ,
            #                         var = 'sig_app', idx = 0,
            #                         record_on = 'update'),
            #                     RTraceDomainListField(name = 'Displacement' ,
            #                                    var = 'u', idx = 1),
            RTraceDomainListField(name='Stress',
                                  #position = 'int_pnts',
                                  var='sig_app', idx=1),
            #                             RTraceDomainListField(name = 'N0' ,
            #                                          var = 'N_mtx', idx = 0,
            #                                          record_on = 'update')
        ])

    # Add the time-loop control
    #
    tl = TLoop(tstepper=ts,
               tline=TLine(min=0.0, step=1., max=1.0))

    tl.eval()
    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tl)
    app.main()
