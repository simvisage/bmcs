from ibvpy.fets.fets_eval import FETSEval
from numpy import \
    array, zeros, int_, float_, ix_, dot, linspace, hstack, vstack, arange, \
    identity
from scipy.linalg import \
    inv
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, Interface, \
    Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
    on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property


#-------------------------------------------------------------------------
# FETS2D4T - 4 nodes iso-parametric quadrilateral element (2D, linear, Lagrange family)
#-------------------------------------------------------------------------
class FETS2D4Q4T(FETSEval):

    debug_on = True

    # Dimensional mapping
    dim_slice = slice(0, 2)

    # Order of node positions for the formulation of shape function
    #
    dof_r = Array(value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])
    geo_r = Array(value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])

    n_e_dofs = Int(4)
    t = Float(1.0, label='thickness')

    # Integration parameters
    #
    ngp_r = Int(2)
    ngp_s = Int(2)

    # Corner nodes are used for visualization
    vtk_r = Array(value=[[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    vtk_cells = [[0, 1, 2, 3]]
    vtk_cell_types = 'Quad'

    # vtk_point_ip_map = [0,1,3,2]
    n_nodal_dofs = Int(1)

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------
    def get_N_geo_mtx(self, r_pnt):
        '''
        Return the value of shape functions for the specified local coordinate r
        '''
        cx = array(self.geo_r, dtype='float_')
        Nr = array([[1 / 4. * (1 + r_pnt[0] * cx[i, 0]) * (1 + r_pnt[1] * cx[i, 1])
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
        cx = array(self.geo_r, dtype='float_')
        dNr_geo = array([[1 / 4. * cx[i, 0] * (1 + r_pnt[1] * cx[i, 1]) for i in range(0, 4)],
                         [1 / 4. * cx[i, 1] * (1 + r_pnt[0] * cx[i, 0]) for i in range(0, 4)]])
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
        I_mtx = identity(self.n_nodal_dofs, float)
        N_mtx_list = [I_mtx * Nr_geo[0, i] for i in range(0, Nr_geo.shape[1])]
        N_mtx = hstack(N_mtx_list)
        return N_mtx

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions
        '''
        return self.get_dNr_geo_mtx(r_pnt)

    def get_B_mtx(self, r_pnt, X_mtx):
        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_mtx = self.get_dNr_mtx(r_pnt)
        dNx_mtx = dot(inv(J_mtx), dNr_mtx)
        Bx_mtx = zeros((2, 4), dtype='float_')
        for i in range(0, 4):
            Bx_mtx[0, i] = dNx_mtx[0, i]
            Bx_mtx[1, i] = dNx_mtx[1, i]
        return Bx_mtx

#----------------------- example --------------------


def run_example():
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, \
        RTraceDomainListInteg, TLoop, \
        TLine, BCDof, IBVPSolve as IS, DOTSEval
    from ibvpy.mats.mats2D.mats2D_conduction.mats2D_conduction import MATS2DConduction

    from ibvpy.api import BCDofGroup
    fets_eval = FETS2D4Q4T(mats_eval=MATS2DConduction(k=1.))

    print(fets_eval.vtk_node_cell_data)

    from ibvpy.mesh.fe_grid import FEGrid
    from ibvpy.mesh.fe_refinement_grid import FERefinementGrid
    from ibvpy.mesh.fe_domain import FEDomain
    from mathkit.mfn import MFnLineArray

    # Discretization
    fe_grid = FEGrid(coord_max=(1., 1., 0.),
                     shape=(2, 2),
                     fets_eval=fets_eval)

    tstepper = TS(sdomain=fe_grid,
                  bcond_list=[BCDofGroup(var='u', value=0., dims=[0],
                                         get_dof_method=fe_grid.get_left_dofs),
                              #                                   BCDofGroup( var='u', value = 0., dims = [1],
                              # get_dof_method = fe_grid.get_bottom_dofs ),
                              BCDofGroup(var='u', value=.005, dims=[0],
                                             get_dof_method=fe_grid.get_top_right_dofs)],
                  rtrace_list=[
                      #                     RTraceDomainListField(name = 'Damage' ,
                      #                                    var = 'omega', idx = 0,
                      #                                    record_on = 'update',
                      #                                    warp = True),
                      #                     RTraceDomainListField(name = 'Displacement' ,
                      #                                    var = 'u', idx = 0,
                      #                                    record_on = 'update',
                      #                                    warp = True),
                      #                    RTraceDomainListField(name = 'N0' ,
                      #                                      var = 'N_mtx', idx = 0,
                      # record_on = 'update')
                  ]
                  )

    # Add the time-loop control
    tloop = TLoop(tstepper=tstepper, debug=False,
                  tline=TLine(min=0.0, step=1.0, max=1.0))

    tloop.eval()
    # Put the whole thing into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    # from ibvpy.plugins.ibvpy_app import IBVPyApp
    # app = IBVPyApp( ibv_resource = tloop )
    # app.main()


if __name__ == '__main__':
    run_example()
