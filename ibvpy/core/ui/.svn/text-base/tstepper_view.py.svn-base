
######### 
#

from enthought.traits.api import \
    Instance, Array, Int, on_trait_change, Property, cached_property, \
    List, Button, HasTraits, implements, WeakRef, Float, Delegate, \
    Callable, Class

import enthought.traits.has_traits
enthought.traits.has_traits.CHECK_INTERFACES = 2

if __name__ == '__main__':    
    from ibvpy.api import \
        TStepper as TS, RTraceGraph, RTraceDomainField, TLoop, \
        TLine, BCDofGroup, BCDof, IBVPSolve as IS
    from ibvpy.rtrace.rt_domain_list_field import RTraceDomainListField
    from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
    from ibvpy.mesh.fe_grid import FEGrid
    from ibvpy.mesh.fe_refinement_level_grid import FERefinementLevelGrid
    from ibvpy.mesh.fe_domain_list import FEDomainList
    
    def example_2d():
        from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
        from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
        
        fets_eval = FETS2D4Q(mats_eval = MATS2DElastic( E = 2.1e5 ))
    
        # Discretization
        fe_domain1 = FEGrid( coord_max = (2.,5.,0.),
                                   shape   = (10,10),
                                   fets_eval = fets_eval )

        fe_subgrid1 = FERefinementLevelGrid( parent_domain = fe_domain1,
                                       fine_cell_shape = (1,1) )

        print 'children'
        print fe_domain1.children
        
        fe_subgrid1.refine_elem( (5,5) )
        fe_subgrid1.refine_elem( (6,5) )
        fe_subgrid1.refine_elem( (7,5) )
        fe_subgrid1.refine_elem( (8,5) )
        fe_subgrid1.refine_elem( (9,5) )
    
        fe_domain  = FEDomainList( subdomains = [ fe_domain1 ] )

        ts = TS( dof_resultants = True,
                 sdomain = fe_domain,
                 bcond_list =  [BCDofGroup(var='f', value = 0.1, dims = [0],
                                           get_dof_method = fe_domain1.get_top_dofs ),
                                BCDofGroup(var='u', value = 0., dims = [0,1],
                                           get_dof_method = fe_domain1.get_bottom_dofs ),
                                           ],
                 rtrace_list =  [ RTraceGraph(name = 'Fi,right over u_right (iteration)' ,
                                       var_y = 'F_int', idx_y = 0,
                                       var_x = 'U_k', idx_x = 1),
                            RTraceDomainListField(name = 'Stress' ,
                                 var = 'sig_app', idx = 0, warp = True ),
#                           RTraceDomainField(name = 'Displacement' ,
    #                                        var = 'u', idx = 0),
    #                                 RTraceDomainField(name = 'N0' ,
    #                                              var = 'N_mtx', idx = 0,
    #                                              record_on = 'update')
                              
                        ]             
                    )

        # Add the time-loop control
        tloop = TLoop( tstepper = ts,
                       tline  = TLine( min = 0.0,  step = 1, max = 1.0 ))
#        
        print tloop.eval()
        from ibvpy.plugins.ibvpy_app import IBVPyApp
        ibvpy_app = IBVPyApp( ibv_resource = tloop )
        ibvpy_app.main()
#            
#    #    print ts.F_int
#    #    print ts.rtrace_list[0].trace.ydata

    example_2d()