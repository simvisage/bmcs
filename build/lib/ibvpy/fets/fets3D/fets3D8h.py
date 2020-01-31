from numpy import \
    zeros, dot, hstack, identity
from scipy.linalg import \
    inv
from traits.api import \
    Instance, Int, Property, Array, cached_property

from .fets3D import FETS3D


#-------------------------------------------------------------------------
# FETS3D8H - 8 nodes isoparametric volume element (3D, linear, Lagrange family)
#-------------------------------------------------------------------------
class FETS3D8H(FETS3D):
    '''
    eight nodes volume element
    '''
    debug_on = False

    # number of nodal degrees of freedom
    # number of degrees of freedom of each element
    n_nodal_dofs = Int(3)
    n_e_dofs = Int(8 * 3)

    # Integration parameters
    #
    ngp_r = 2
    ngp_s = 2
    ngp_t = 2

    # Here an isoparametric element formulation is applied.
    # The implemented shape functions are derived based on the following
    # ordering of the nodes of the parent element

    dof_r = \
        Array(value=[[-1., -1., -1.],
                     [1., -1., -1.],
                     [-1., 1., -1.],
                     [1., 1., -1.],
                     [-1., -1., 1.],
                     [1., -1., 1.],
                     [-1., 1., 1.],
                     [1., 1., 1.]])

    geo_r = \
        Array(value=[[-1., -1., -1.],
                     [1., -1., -1.],
                     [-1., 1., -1.],
                     [1., 1., -1.],
                     [-1., -1., 1.],
                     [1., -1., 1.],
                     [-1., 1., 1.],
                     [1., 1., 1.]])

    # Used for Visualization
    vtk_r = Array(value=[[-1., -1., -1.],
                         [1., -1., -1.],
                         [-1., 1., -1.],
                         [1., 1., -1.],
                         [-1., -1., 1.],
                         [1., -1., 1.],
                         [-1., 1., 1.],
                         [1., 1., 1.]])
    vtk_cells = [[0, 1, 3, 2, 4, 5, 7, 6]]
    vtk_cell_types = 'Hexahedron'

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------
    def get_N_geo_mtx(self, r_pnt):
        '''
        Return the value of shape functions (derived in femple) for the
        specified local coordinate r_pnt
        '''
        N_geo_mtx = zeros((1, 8), dtype='float_')
        N_geo_mtx[0, 0] = -((-1 + r_pnt[2]) * (-1 + r_pnt[1]) *
                            (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 1] = ((-1 + r_pnt[2]) * (-1 + r_pnt[1]) *
                           (1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 2] = ((-1 + r_pnt[2]) * (1 + r_pnt[1]) *
                           (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 3] = -((-1 + r_pnt[2]) * (1 + r_pnt[1]) *
                            (1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 4] = ((1 + r_pnt[2]) * (-1 + r_pnt[1]) *
                           (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 5] = -((1 + r_pnt[2]) * (-1 + r_pnt[1]) *
                            (1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 6] = -((1 + r_pnt[2]) * (1 + r_pnt[1]) *
                            (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 7] = ((1 + r_pnt[2]) * (1 + r_pnt[1]) *
                           (1 + r_pnt[0])) / 8.0
        return N_geo_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives (derived in femple).
        Used for the construction of the Jacobi matrix.
        '''
        dNr_geo_mtx = zeros((3, 8), dtype='float_')
        dNr_geo_mtx[0, 0] = -((-1 + r_pnt[2]) * (-1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 1] = ((-1 + r_pnt[2]) * (-1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 2] = ((-1 + r_pnt[2]) * (1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 3] = -((-1 + r_pnt[2]) * (1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 4] = ((1 + r_pnt[2]) * (-1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 5] = -((1 + r_pnt[2]) * (-1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 6] = -((1 + r_pnt[2]) * (1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 7] = ((1 + r_pnt[2]) * (1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[1, 0] = -((-1 + r_pnt[2]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 1] = ((-1 + r_pnt[2]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 2] = ((-1 + r_pnt[2]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 3] = -((-1 + r_pnt[2]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 4] = ((1 + r_pnt[2]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 5] = -((1 + r_pnt[2]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 6] = -((1 + r_pnt[2]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 7] = ((1 + r_pnt[2]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 0] = -((-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 1] = ((-1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 2] = ((1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 3] = -((1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 4] = ((-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 5] = -((-1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 6] = -((1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 7] = ((1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        return dNr_geo_mtx

    #---------------------------------------------------------------------
    # Method delivering the shape functions for the field variables and their
    # derivatives
    #---------------------------------------------------------------------
    def get_N_mtx(self, r_pnt):
        '''
        Returns the matrix of the shape functions used for the field approximation
        containing zero entries. The number of rows corresponds to the number of nodal
        dofs. The matrix is evaluated for the specified local coordinate r_pnt.
        '''
        N = self.get_N_geo_mtx(r_pnt)
        I_mtx = identity(self.n_nodal_dofs, float)
        N_mtx_list = [I_mtx * N[0, i] for i in range(0, N.shape[1])]
        N_mtx = hstack(N_mtx_list)
        return N_mtx

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions used for the field approximation
        '''
        dNr_mtx = self.get_dNr_geo_mtx(r_pnt)
        return dNr_mtx

#----------------------- example --------------------


if __name__ == '__main__':
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine, BCDofGroup, IBVPSolve as IS, DOTSEval

    from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import MATS3DElastic

    fets_eval = FETS3D8H(mats_eval=MATS3DElastic(nu=0.25))

    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid(coord_max=(3., 3., 3.),
                    shape=(10, 10, 10),
                    fets_eval=fets_eval)

    ts = TS(
        sdomain=domain,
        bcond_list=[BCDofGroup(var='u', value=0., dims=[0],
                               get_dof_method=domain.get_left_dofs),
                    BCDofGroup(var='u', value=0., dims=[1, 2],
                               get_dof_method=domain.get_bottom_left_dofs),
                    BCDofGroup(var='u', value=0.002, dims=[0],
                               get_dof_method=domain.get_right_dofs)],
        rtrace_list=[
            #                        RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
            #                                  var_y = 'F_int', idx_y = right_dof,
            #                                  var_x = 'U_k', idx_x = right_dof,
            #                                  record_on = 'update'),
            RTraceDomainListField(name='Deformation',
                                       var='eps_app', idx=0,
                                       record_on='update'),
            RTraceDomainListField(name='Displacement',
                                       var='u', idx=0),
            #                         RTraceDomainListField(name = 'Stress' ,
            #                                        var = 'sig', idx = 0,
            #                                        record_on = 'update'),
            #                        RTraceDomainListField(name = 'N0' ,
            #                                       var = 'N_mtx', idx = 0,
            #                                       record_on = 'update')
        ]
    )

    # Add the time-loop control
    tloop = TLoop(tstepper=ts,
                  tline=TLine(min=0.0, step=0.5, max=1.0))

    tloop.eval()

    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tloop)
    app.main()
