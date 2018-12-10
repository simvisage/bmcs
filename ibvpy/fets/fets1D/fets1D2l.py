
from numpy import array, dot
from scipy.linalg import \
    inv

from traits.api import \
    Int, Array
from ibvpy.fets.fets_eval import FETSEval


#-----------------------------------------------------------------------------
# FEBar1D
#-----------------------------------------------------------------------------
class FETS1D2L(FETSEval):
    '''
    Fe Bar 2 nodes, deformation
    '''

    debug_on = True

    # Dimensional mapping
    dim_slice = slice(0, 1)

    n_e_dofs = Int(2)
    n_nodal_dofs = Int(1)

    dof_r = Array(value=[[-1], [1]])
    geo_r = Array(value=[[-1], [1]])
    vtk_r = Array(value=[[-1.], [1.]])
    vtk_cells = [[0, 1]]
    vtk_cell_types = 'Line'

    def _get_ip_coords(self):
        offset = 1e-6
        return array([[-1 + offset, 0., 0.], [1 - offset, 0., 0.]])

    def _get_ip_weights(self):
        return array([[1.], [1.]], dtype=float)

    # Integration parameters
    #
    ngp_r = 2

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
        '''
        return array([[-1. / 2, 1. / 2]])

    def get_N_mtx(self, r_pnt):
        '''
        Return shape functions
        @param r_pnt:local coordinates
        '''
        return self.get_N_geo_mtx(r_pnt)

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions
        '''
        return self.get_dNr_geo_mtx(r_pnt)

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


from ibvpy.plugins.ibvpy_app import IBVPyApp


def __demo__():

    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine, BCDof
    from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic

    fets_eval = FETS1D2L(mats_eval=MATS1DElastic(E=10.))
    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid(coord_max=(3.,),
                    shape=(3,),
                    fets_eval=fets_eval)

    ts = TS(dof_resultants=True,
            sdomain=domain,
            bcond_list=[BCDof(var='u', dof=0, value=0.),
                        BCDof(var='f', dof=3, value=1,)],
            rtrace_list=[RTDofGraph(name='Fi,right over u_right (iteration)',
                                    var_y='F_int', idx_y=0,
                                    var_x='U_k', idx_x=1),
                         RTraceDomainListField(name='Stress',
                                               var='sig_app', idx=0),
                         RTraceDomainListField(name='Displacement',
                                               var='u', idx=0,
                                               warp=True),
                         RTraceDomainListField(name='N0',
                                               var='N_mtx', idx=0,
                                               record_on='update')
                         ]
            )

    # Add the time-loop control
    tloop = TLoop(tstepper=ts,
                  tline=TLine(min=0.0, step=0.5, max=1.0))

    print('---- result ----')
    print(tloop.eval())
    print(ts.F_int)
    print(ts.rtrace_list[0].trace.ydata)

    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    app = IBVPyApp(ibv_resource=tloop)
    app.main()


if __name__ == '__main__':
    __demo__()
