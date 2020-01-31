'''
Created on Jun 18, 2009

@author: jakub
'''
from numpy import array, dot
from scipy.linalg import \
    inv
from traits.api import \
    Int, Array
from ibvpy.fets.fets_eval import FETSEval


#-----------------------------------------------------------------------------
# FEBar1D
#-----------------------------------------------------------------------------
class FETS1D2L3U(FETSEval):
    '''
    Fe Bar 3 nodes, deformation
    '''

    debug_on = True

    # Dimensional mapping
    dim_slice = slice(0, 1)

    n_e_dofs = Int(3)
    n_nodal_dofs = Int(1)

    dof_r = Array(value=[[-1], [0], [1]])
    geo_r = Array(value=[[-1], [1]])

    vtk_r = Array(value=[[-1.], [0.], [1.]])
    vtk_cells = [[0, 2, 1]]
    vtk_cell_types = 'QuadraticEdge'

    # Integration parameters
    #
    ngp_r = 3

    def get_N_geo_mtx(self, r_pnt):
        '''
        Return geometric shape functions
        @param r_pnt:
        '''
        r = r_pnt[0]
        N_mtx = array([[0.5 - r / 2., 0.5 + r / 2.]])
        return N_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives.
        Used for the conrcution of the Jacobi matrix.

        @TODO - the B matrix is used
        just for uniaxial bar here with a trivial differential
        operator.
        '''
        return array([[-1. / 2, 1. / 2]])

    def get_N_mtx(self, r_pnt):
        '''
        Return shape functions
        @param r_pnt:local coordinates
        '''
        r = r_pnt[0]
        return array([[1. / 2 * r * (r - 1.), 1. - r * r, 1. / 2 * r * (r + 1.)]])

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions
        '''
        r = r_pnt[0]
        return array([[r - 1. / 2., -2. * r, r + 1. / 2.]])

    def get_B_mtx(self, r, X):
        '''
        Return kinematic matrix
        @param r:local coordinates
        @param X:global coordinates
        '''
        J_mtx = self.get_J_mtx(r, X)
        dNr_mtx = self.get_dNr_mtx(r)
        B_mtx = dot(inv(J_mtx), dNr_mtx)
        return B_mtx

#----------------------- example --------------------


def example_with_new_domain():
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine, BCDof, IBVPSolve as IS, DOTSEval
    from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic

    fets_eval = FETS1D2L3U(mats_eval=MATS1DElastic(E=10.))

    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid(coord_max=(3., ),
                    shape=(3, ),
                    fets_eval=fets_eval)

    ts = TS(dof_resultants=True,
            sdomain=domain,
            # conversion to list (square brackets) is only necessary for slicing of
            # single dofs, e.g "get_left_dofs()[0,1]"
            #         bcond_list =  [ BCDof(var='u', dof = 0, value = 0.)     ] +
            #                    [ BCDof(var='u', dof = 2, value = 0.001 ) ]+
            #                    [ )     ],
            bcond_list=[BCDof(var='u', dof=0, value=0.),
                        #                        BCDof(var='u', dof = 1, link_dofs = [2], link_coeffs = [0.5],
                        #                              value = 0. ),
                        #                        BCDof(var='u', dof = 2, link_dofs = [3], link_coeffs = [1.],
                        #                              value = 0. ),
                        BCDof(var='f', dof=6, value=1,
                              # link_dofs = [2], link_coeffs = [2]
                              )],
            rtrace_list=[RTDofGraph(name='Fi,right over u_right (iteration)',
                                    var_y='F_int', idx_y=0,
                                    var_x='U_k', idx_x=1),
                         RTraceDomainListField(name='Stress',
                                               var='sig_app', idx=0),
                         RTraceDomainListField(name='Displacement',
                                               var='u', idx=0),
                         RTraceDomainListField(name='N0',
                                               var='N_mtx', idx=0,
                                               record_on='update')

                         ]
            )

    # Add the time-loop control
    tloop = TLoop(tstepper=ts,
                  tline=TLine(min=0.0, step=1, max=1.0))

    print('---- result ----')
    print(tloop.eval())
    print(ts.F_int)
    print(ts.rtrace_list[0].trace.ydata)

    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tloop)
    app.main()


if __name__ == '__main__':
    example_with_new_domain()
