from enthought.traits.api import \
    Int, implements

from ibvpy.fets.fets_eval import IFETSEval, FETSEval


from numpy import array, dot

from scipy.linalg import \
     inv

#-----------------------------------------------------------------------------
# FEBar1D
#-----------------------------------------------------------------------------

class FETS1D2L( FETSEval ):
    '''
    Fe Bar 2 nodes, deformation
    '''

    implements( IFETSEval )

    debug_on = True

    # Dimensional mapping
    dim_slice = slice( 0, 1 )

    n_e_dofs = Int( 2 )
    n_nodal_dofs = Int( 1 )

    dof_r = [[-1], [1]]
    geo_r = [[-1], [1]]
    vtk_r = [[-1.], [1.]]
    vtk_cells = [[0, 1]]
    vtk_cell_types = 'Line'

    def _get_ip_coords( self ):
        offset = 1e-6
        return  array( [[-1 + offset, 0., 0.], [1 - offset, 0., 0.]] )

    def _get_ip_weights( self ):
        return array( [[1.], [1.]], dtype = float )


    # Integration parameters
    #
    ngp_r = 2

    def get_N_geo_mtx( self, r_pnt ):
        '''
        Return geometric shape functions
        @param r_pnt:
        '''
        r = r_pnt[0]
        N_mtx = array( [[0.5 - r / 2., 0.5 + r / 2.]] )
        return N_mtx

    def get_dNr_geo_mtx( self, r_pnt ):
        '''
        Return the matrix of shape function derivatives.
        Used for the conrcution of the Jacobi matrix.
        '''
        return array( [[-1. / 2, 1. / 2]] )

    def get_N_mtx( self, r_pnt ):
        '''
        Return shape functions
        @param r_pnt:local coordinates
        '''
        return self.get_N_geo_mtx( r_pnt )

    def get_dNr_mtx( self, r_pnt ):
        '''
        Return the derivatives of the shape functions
        '''
        return self.get_dNr_geo_mtx( r_pnt )

    def get_B_mtx( self, r, X ):
        '''
        Return kinematic matrix
        @param r:local coordinates
        @param X:global coordinates
        '''
        J_mtx = self.get_J_mtx( r, X )
        dNr_mtx = self.get_dNr_mtx( r )
        B_mtx = dot( inv( J_mtx ), dNr_mtx )
        return B_mtx

#----------------------- example --------------------

def example_with_new_domain():
    from ibvpy.api import \
        TStepper as TS, RTraceGraph, RTraceDomainListField, TLoop, \
        TLine, BCDof, IBVPSolve as IS, DOTSEval
    from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
    from ibvpy.mats.mats1D.mats1D_damage.mats1D_damage import MATS1DDamage

    fets_eval = FETS1D2L( mats_eval = MATS1DElastic( E = 10. ) )
    #fets_eval = FETS1D2L(mats_eval = MATS1DDamage()) 
    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid( coord_max = ( 3., 0., 0. ),
                           shape = ( 3, ),
                           fets_eval = fets_eval )

    ts = TS( dof_resultants = True,
             sdomain = domain,
         # conversion to list (square brackets) is only necessary for slicing of 
         # single dofs, e.g "get_left_dofs()[0,1]"
#         bcond_list =  [ BCDof(var='u', dof = 0, value = 0.)     ] +  
#                    [ BCDof(var='u', dof = 2, value = 0.001 ) ]+
#                    [ )     ],
         bcond_list = [BCDof( var = 'u', dof = 0, value = 0. ),
#                        BCDof(var='u', dof = 1, link_dofs = [2], link_coeffs = [0.5],
#                              value = 0. ),
#                        BCDof(var='u', dof = 2, link_dofs = [3], link_coeffs = [1.],
#                              value = 0. ),
                        BCDof( var = 'f', dof = 3, value = 1,
                                  #link_dofs = [2], link_coeffs = [2]
                                   ) ],
         rtrace_list = [ RTraceGraph( name = 'Fi,right over u_right (iteration)' ,
                               var_y = 'F_int', idx_y = 0,
                               var_x = 'U_k', idx_x = 1 ),
                    RTraceDomainListField( name = 'Stress' ,
                         var = 'sig_app', idx = 0 ),
                     RTraceDomainListField( name = 'Displacement' ,
                                    var = 'u', idx = 0,
                                    warp = True ),
                             RTraceDomainListField( name = 'N0' ,
                                          var = 'N_mtx', idx = 0,
                                          record_on = 'update' )

                ]
            )

    # Add the time-loop control
    tloop = TLoop( tstepper = ts,
                   tline = TLine( min = 0.0, step = 0.5, max = 1.0 ) )

    print '---- result ----'
    print tloop.eval()
    print ts.F_int
    print ts.rtrace_list[0].trace.ydata

    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp( ibv_resource = tloop )
    app.main()


if __name__ == '__main__':
    example_with_new_domain()
