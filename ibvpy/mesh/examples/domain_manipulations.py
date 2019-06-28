

from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
from ibvpy.fets.fets2D.fets2D4q8u import FETS2D4Q8U
from ibvpy.api import\
     BCDofGroup, TStepper as TS, TLoop, TLine, RTDofGraph
from ibvpy.rtrace.rt_domain_list_field import RTraceDomainListField
from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.mesh.fe_refinement_grid import FERefinementGrid
from ibvpy.mesh.fe_domain import FEDomain

if __name__ == '__main__':

    fets_eval_4u = FETS2D4Q(mats_eval = MATS2DElastic())
    fets_eval_8u = FETS2D4Q8U(mats_eval = MATS2DElastic())
        
    fe_domain = FEDomain()

    fe_rgrid1 = FERefinementGrid( name = 'fe_rgrid1', fets_eval = fets_eval_4u, domain = fe_domain )

    fe_grid1 = FEGrid( name = 'fe_grid1', coord_max = (2.,6.,0.), 
                               shape   = (1,3),
                               fets_eval = fets_eval_4u,
                               level = fe_rgrid1 )    

    fe_grid2 = FEGrid( name = 'fe_grid2', coord_min = (2.,  6, 0.),
                      coord_max = (10, 15, 0.), 
                               shape   = (1,3),
                               fets_eval = fets_eval_4u,
                               level = fe_rgrid1 )    
        
    fe_rgrid2 = FERefinementGrid( name = 'fe_rgrid2', fets_eval = fets_eval_4u, domain = fe_domain )
    
    fe_grid3 = FEGrid( name = 'fe_grid3', coord_min = (0, 0, 1.),
                      coord_max = (2., 6.,1.), 
                               shape   = (1,3),
                               fets_eval = fets_eval_4u,
                               level = fe_rgrid2 )    

    fe_grid4 = FEGrid( name = 'fe_grid4', coord_min = (2.,  6, 1.),
                      coord_max = (10, 15, 1.), 
                               shape   = (1,3),
                               fets_eval = fets_eval_4u,
                               level = fe_rgrid2 )        

    fe_rgrid3 = FERefinementGrid( name = 'fe_rgrid3', fets_eval = fets_eval_4u, domain = fe_domain )
    
    fe_grid5 = FEGrid( name = 'fe_grid5', coord_min = (0, 0, 2.),
                      coord_max = (2., 6.,2.), 
                               shape   = (1,3),
                               fets_eval = fets_eval_4u,
                               level = fe_rgrid3 )                                   

    fe_grid6 = FEGrid( name = 'fe_grid6', coord_min = (2.,  6, 2.),
                      coord_max = (10, 15, 2.), 
                               shape   = (1,3),
                               fets_eval = fets_eval_4u,
                               level = fe_rgrid3 )                                   

    ts = TS( dof_resultants = True,
             sdomain = fe_domain,
             bcond_list =  [BCDofGroup(var='f', value = 1., dims = [0],
                                       get_dof_method = fe_grid1.get_top_dofs ),
                            BCDofGroup(var='u', value = 0., dims = [0,1],
                                       get_dof_method = fe_grid1.get_bottom_dofs ),
                                       ],
             rtrace_list =  [ RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
                                   var_y = 'F_int', idx_y = 0,
                                   var_x = 'U_k', idx_x = 1),
                        RTraceDomainListField(name = 'Stress',
                             var = 'sig_app', idx = 0, warp = False ),
#                             RTraceDomainField(name = 'Displacement' ,
#                                        var = 'u', idx = 0),
#                                 RTraceDomainField(name = 'N0' ,
#                                              var = 'N_mtx', idx = 0,
#                                              record_on = 'update')
                    ]
                )
    
    
    # Add the time-loop control
    tloop = TLoop( tstepper = ts,
                   tline  = TLine( min = 0.0,  step = 1, max = 1.0 ))
    
    print(tloop.setup())
    
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp( ibv_resource = tloop )
    ibvpy_app.main()
        