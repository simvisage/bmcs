

from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.mats.mats2D.mats2D_sdamage import MATS2DScalarDamage
from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
from ibvpy.fets.fets2D.fets2D4q8u import FETS2D4Q8U
from ibvpy.fets.fets_ls.fets_ls_eval import FETSLSEval #, FETSCracked
from ibvpy.api import\
     BCDofGroup, TStepper as TS, TLoop, TLine, RTDofGraph
from ibvpy.rtrace.rt_domain_list_field import RTraceDomainListField
from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.mesh.fe_refinement_grid import FERefinementGrid

def notched_bended_beam():

    fets_eval_4u      = FETS2D4Q( mats_eval = MATS2DScalarDamage() )
    fets_eval_cracked = FETSLSEval( parent_fets  = fets_eval_4u )

    # Discretization
    fe_domain1 = FEGrid( coord_max = (5.,2.,0.), 
                               shape   = (3,2),
                               fets_eval = fets_eval_4u )

    fe_child_domain = FERefinementGrid( parent_domain = fe_domain1,
                                    fets_eval = fets_eval_cracked,
                                    fine_cell_shape = (1,1) )

    crack_level_set = lambda X: X[0] - 2.5  
    
    fe_child_domain.refine_elem( (1,0), crack_level_set )
    dots = fe_child_domain.new_dots()

    fe_domain  = FEDomainList( subdomains = [ fe_domain1 ] )
    fe_domain_tree = FEDomainTree( domain_list = fe_domain )
    
    ts = TS( dof_resultants = True,
             sdomain = [ fe_domain1, fe_child_domain ],
             bcond_list =  [BCDofGroup(var='u', value = 0., dims = [0,1],
                                       get_dof_method = fe_domain1.get_left_dofs ),
                            BCDofGroup(var='u', value = 0., dims = [0,1],
                                       get_dof_method = fe_domain1.get_right_dofs ),
                            BCDofGroup(var='f', value = -1., dims = [1],
                                       get_dof_method = fe_domain1.get_top_dofs ),
                                       ],
             rtrace_list =  [
#                              RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
#                                   var_y = 'F_int', idx_y = 0,
#                                   var_x = 'U_k', idx_x = 1),
#                        RTraceDomainListField(name = 'Stress' ,
#                             var = 'sig_app', idx = 0, warp = True ),
#                             RTraceDomainField(name = 'Displacement' ,
#                                        var = 'u', idx = 0),
#                                 RTraceDomainField(name = 'N0' ,
#                                              var = 'N_mtx', idx = 0,
#                                              record_on = 'update')
#                          
                    ]             
                )
    
    # Add the time-loop control
    tloop = TLoop( tstepper = ts,
                   tline  = TLine( min = 0.0,  step = 1, max = 1.0 ))
    
    print(tloop.eval())
#    from ibvpy.plugins.ibvpy_app import IBVPyApp
#    ibvpy_app = IBVPyApp( ibv_resource = tloop )
#    ibvpy_app.main()
        
#    print ts.F_int
#    print ts.rtrace_list[0].trace.ydata

if __name__ == '__main__':
    notched_bended_beam()

    