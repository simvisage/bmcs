

from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
from ibvpy.fets.fets2D.fets2D4q8u import FETS2D4Q8U
from ibvpy.api import\
     BCSlice, TStepper as TS, TLoop, TLine, RTDofGraph
from ibvpy.rtrace.rt_domain_list_field import RTraceDomainListField
from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.mesh.fe_refinement_grid import FERefinementGrid
from ibvpy.mesh.fe_domain import FEDomain

def combined_fe2D4q_with_fe2D4q8u():

    fets_eval_4u_conc = FETS2D4Q( mats_eval = MATS2DElastic( E = 28500, nu = 0.2 ) )
    fets_eval_4u_steel = FETS2D4Q( mats_eval = MATS2DElastic( E = 210000, nu = 0.25 ) )
    fets_eval_8u = FETS2D4Q8U( mats_eval = MATS2DElastic() )

    # Discretization
    fe_domain = FEDomain()

    fe_grid_level1 = FERefinementGrid( name = 'master grid',
                                       fets_eval = fets_eval_4u_conc,
                                       domain = fe_domain )

    fe_grid = FEGrid( level = fe_grid_level1,
                      coord_max = ( 2., 6., 0. ),
                      shape = ( 11, 30 ),
                      fets_eval = fets_eval_4u_conc )

    fe_grid_level2 = FERefinementGrid( name = 'refinement grid',
                                       parent = fe_grid_level1,
                                       fets_eval = fets_eval_4u_steel,
                                       fine_cell_shape = ( 1, 1 ) )

    # fe_grid_level1[ 5, :5 ].refine_using( fe_grid_level2 )
    # 1. first get the slice for the level - distinguish it from the slice at the subgrid
    #    this includes slicing in the subgrids. what if the subgrid does not exist?
    #    
    #    Each subgrid must hold its own slice within the level. The index operator fills
    #    the grid [...] instanciates the whole grid and returns the instance of 
    #    FEGridLevelSlice. The expanded subgrid contains its constructor slice.
    #
    # 2. If the slice is within an existing slice no change in the FESubgrid is required
    #    only the instance of the slice is returned. The FEGridLevelSlice goes always into 
    #    an expanded part of FEGrid.
    #
    # 3. If the slice does not fit into any existing slice - all domain with an intersection
    #    of the existing slice must be constructed as well. 
    #
    # 2. deactivate elements
    # 3.
    # BUT how to impose the boundary conditions on the particular refinement? The
    # slice has an attribute  

    fe_grid_level2.refine_elem( ( 5, 0 ) )
    fe_grid_level2.refine_elem( ( 5, 1 ) )
    fe_grid_level2.refine_elem( ( 5, 2 ) )
    fe_grid_level2.refine_elem( ( 5, 3 ) )
    fe_grid_level2.refine_elem( ( 5, 4 ) )
    fe_grid_level2.refine_elem( ( 5, 5 ) )

    # apply the boundary condition on a subgrid
    #
    print(fe_grid_level2.fe_subgrids)
    fe_first_grid = fe_grid_level2.fe_subgrids[0]

    ts = TS( dof_resultants = True,
             sdomain = fe_domain,
             bcond_list = [BCSlice( var = 'f', value = 1., dims = [0],
                                       slice = fe_grid[ :, -1, :, -1 ] ),
                           BCSlice( var = 'u', value = 0., dims = [0, 1],
                                       slice = fe_first_grid[ :, 0, :, 0 ] )
                                       ],
             rtrace_list = [ RTDofGraph( name = 'Fi,right over u_right (iteration)' ,
                                   var_y = 'F_int', idx_y = 0,
                                   var_x = 'U_k', idx_x = 1 ),
                        RTraceDomainListField( name = 'Stress',
                             var = 'sig_app', idx = 0, warp = True ),
#                             RTraceDomainField(name = 'Displacement' ,
#                                        var = 'u', idx = 0),
#                                 RTraceDomainField(name = 'N0' ,
#                                              var = 'N_mtx', idx = 0,
#                                              record_on = 'update')
                    ]
                )

    # Add the time-loop control
    tloop = TLoop( tstepper = ts,
                   tline = TLine( min = 0.0, step = 1, max = 1.0 ) )

    print(tloop.eval())
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp( ibv_resource = tloop )
    ibvpy_app.main()

#    print ts.F_int
#    print ts.rtrace_list[0].trace.ydata

if __name__ == '__main__':
    combined_fe2D4q_with_fe2D4q8u()
