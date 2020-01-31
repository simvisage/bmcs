
#from sys_matrix import SysSparseMtx, SysDenseMtx
from numpy import array, zeros, arange, array_equal, hstack, dot, sqrt
from scipy.linalg import norm

from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
from mathkit.matrix_la.coo_mtx import COOSparseMtx
from mathkit.matrix_la.dense_mtx import DenseMtx
import unittest

from ibvpy.api import \
    TStepper as TS, RTDofGraph, RTraceDomainField, TLoop, \
    TLine, BCDof, BCDofGroup, IBVPSolve as IS

from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.rtrace.rt_domain_list_field import \
     RTraceDomainListField 

from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
from ibvpy.fets.fets1D.fets1D2l import FETS1D2L

from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm import MATS2DMicroplaneDamage
from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
from ibvpy.dots.dots_eval import DOTSEval
from ibvpy.dots.dots_list_eval import DOTSListEval

def test_bar2( ):
    '''Clamped bar composed of two linked bars loaded at the right end
    [00]-[01]-[02]-[03]-[04]-[05]-[06]-[07]-[08]-[09]-[10]
    [11]-[12]-[13]-[14]-[15]-[16]-[17]-[18]-[19]-[20]-[21]
    u[0] = 0, u[5] = u[16], R[-1] = R[21] = 10
    '''
    fets_eval = FETS1D2L(mats_eval = MATS1DElastic(E=10., A=1.))        

    # Discretization
    fe_domain1 = FEGrid( coord_max = (10.,0.,0.), 
                                    shape   = (10,),
                                    n_nodal_dofs = 1,
                                    dof_r = fets_eval.dof_r,
                                    geo_r = fets_eval.geo_r )

    fe_domain2 = FEGrid( coord_min = (10.,0.,0.),  
                               coord_max = (20.,0.,0.), 
                               shape   = (10,),
                               n_nodal_dofs = 1,
                               dof_r = fets_eval.dof_r,
                               geo_r = fets_eval.geo_r )

    ts = TS( iterms = [ ( fets_eval, fe_domain1 ), (fets_eval, fe_domain2 ) ],
             dof_resultants = True,
             bcond_list =  [BCDof(var='u', dof = 0, value = 0.),
                           BCDof(var='u', dof = 5, link_dofs = [16], link_coeffs = [1.],
                                  value = 0. ),
                           BCDof(var='f', dof = 21, value = 10 ) ],
             rtrace_list =  [ RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
                                   var_y = 'F_int', idx_y = 0,
                                   var_x = 'U_k', idx_x = 1),
                                   ]             
                )

    # Add the time-loop control
    tloop = TLoop( tstepper = ts,
                        tline  = TLine( min = 0.0,  step = 1, max = 1.0 ))    
    u = tloop.eval()
    print('u', u)
    #
    # '---------------------------------------------------------------'
    # 'Clamped bar composed of two linked bars control displ at right'
    # 'u[0] = 0, u[5] = u[16], u[21] = 1'
    # Remove the load and put a unit displacement at the right end
    # Note, the load is irrelevant in this case and will be rewritten
    #
    ts.bcond_list =  [BCDof(var='u', dof = 0, value = 0.),
                      BCDof(var='u', dof = 5, link_dofs = [16], link_coeffs = [1.],
                            value = 0. ),
                            BCDof(var='u', dof = 21, value = 1. ) ]
    # system solver
    u = tloop.eval()
    print('u',u)

def test_bar4( ):
    '''Clamped bar 3 domains, each with 2 elems (displ at right end)
    [0]-[1]-[2] [3]-[4]-[5] [6]-[7]-[8]
    u[0] = 0, u[2] = u[3], u[5] = u[6], u[8] = 1'''

    fets_eval = FETS1D2L(mats_eval = MATS1DElastic(E=10., A=1.))        

    # Discretization
    fe_domain1 = FEGrid( coord_max = (2.,0.,0.), 
                                    shape   = (2,),
                                    n_nodal_dofs = 1,
                                    dof_r = fets_eval.dof_r,
                                    geo_r = fets_eval.geo_r )

    fe_domain2 = FEGrid( coord_min = (2.,0.,0.),  
                               coord_max = (4.,0.,0.), 
                               shape   = (2,),
                               n_nodal_dofs = 1,
                               dof_r = fets_eval.dof_r,
                               geo_r = fets_eval.geo_r )

    fe_domain3 = FEGrid( coord_min = (4.,0.,0.),  
                               coord_max = (6.,0.,0.), 
                               shape   = (2,),
                               n_nodal_dofs = 1,
                               dof_r = fets_eval.dof_r,
                               geo_r = fets_eval.geo_r )
        
    ts = TS( iterms = [ ( fets_eval, fe_domain1 ), (fets_eval, fe_domain2 ), (fets_eval, fe_domain3 ) ], 
             dof_resultants = True,
             bcond_list =  [BCDof(var='u', dof = 0, value = 0.),
                            BCDof(var='u', dof = 2, link_dofs = [3], link_coeffs = [1.],
                                  value = 0. ),
                            BCDof(var='u', dof = 5, link_dofs = [6], link_coeffs = [1.],
                                  value = 0. ),
                            BCDof(var='u', dof = 8, value = 1) ],
             rtrace_list =  [ RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
                                   var_y = 'F_int', idx_y = 0,
                                   var_x = 'U_k', idx_x = 1),
                             RTraceDomainListField( name = 'Displacement', var = 'u', idx = 0 )
                                   ]             
                )
    
    # Add the time-loop control
    tloop = TLoop( tstepper = ts,
                        tline  = TLine( min = 0.0,  step = 1, max = 1.0 ))
    

    print(tloop.eval())
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp( ibv_resource = tloop )
    app.main()      

def xtest_L_shaped( ):
    '''Clamped bar 3 domains, each with 2 elems (displ at right end)
    [0]-[1]-[2] [3]-[4]-[5] [6]-[7]-[8]
    u[0] = 0, u[2] = u[3], u[5] = u[6], u[8] = 1'''
    
    mp = MATS2DScalarDamage(E = 34.e3,
                                   nu = 0.2,
                                   epsilon_0 = 59.e-6,
                                   epsilon_f = 3.2e-3,
                                   #epsilon_f = 3.2e-1,
                                   #stiffness  = "algorithmic",
                                   strain_norm_type = 'Mises')
                                                   
#    mp = MATS2DElastic( E = 34.e3,
#                        nu = 0.2 ) 
    fets_eval = FETS2D4Q(mats_eval = mp ) 

    discr = ( 10, 10 )
    # Discretization
    fe_domain1 = FEGrid( coord_min = (0,0,0),
                               coord_max = (1.,1.,0.), 
                               shape   = discr,
                               n_nodal_dofs = fets_eval.n_nodal_dofs,
                               dof_r = fets_eval.dof_r,
                               geo_r = fets_eval.geo_r )

    fe_domain2 = FEGrid( coord_min = (0.,1.,0),
                               coord_max = (1.,2.,0.), 
                               shape   = discr,
                               n_nodal_dofs = fets_eval.n_nodal_dofs,
                               dof_r = fets_eval.dof_r,
                               geo_r = fets_eval.geo_r )
    
    fe_domain3 = FEGrid( coord_min = (1.,1.,0),
                               coord_max = (2.,2.,0.), 
                               shape   = discr,
                               n_nodal_dofs = fets_eval.n_nodal_dofs,
                               dof_r = fets_eval.dof_r,
                               geo_r = fets_eval.geo_r )

    ts = TS( iterms = [ ( fets_eval, fe_domain1 ),
                        ( fets_eval, fe_domain2 ),
                        ( fets_eval, fe_domain3 ) ],
             dof_resultants = True,
             bcond_list =  [ BCDofGroup( var='u', value = 0., dims = [0,1],
                                         get_dof_method = fe_domain1.get_bottom_dofs ),
                             BCDofGroup( var='u', value = 0., dims = [0,1],
                                         get_dof_method = fe_domain3.get_left_dofs,
                                         get_link_dof_method = fe_domain2.get_right_dofs,
                                         link_coeffs = [1.] ),                                                                        
                             BCDofGroup( var='u', value = 0., dims = [0,1],
                                         get_dof_method = fe_domain2.get_bottom_dofs,
                                         get_link_dof_method = fe_domain1.get_top_dofs,
                                         link_coeffs = [1.] ),
                             BCDofGroup( var='u', value = 0.0004, dims = [1],
                                         get_dof_method = fe_domain3.get_right_dofs ) ],
             rtrace_list =  [ RTraceDomainListField( name = 'Displacement', 
                                                     var = 'u', 
                                                     idx = 1 ),
                              RTraceDomainListField(name = 'Damage' ,
                              var = 'omega', idx = 0,
                              record_on = 'update',
                              warp = True),
#                              RTraceDomainListField(name = 'Stress' ,
#                              var = 'sig_app', idx = 0,
#                              record_on = 'update',
#                              warp = False),
#                              RTraceDomainListField(name = 'Strain' ,
#                              var = 'eps_app', idx = 0,
#                              record_on = 'update',
#                              warp = False), 
                              ]             
            )

    # Add the time-loop control
    global tloop
    tloop = TLoop( tstepper = ts, tolerance = 1e-4, KMAX = 50,
                        tline  = TLine( min = 0.0,  step = 0.2, max = 1.0 ))

    tloop.eval()
#    import cProfile
#    cProfile.run('tloop.eval()', 'tloop_prof' )
#    
#    import pstats
#    p = pstats.Stats('tloop_prof')
#    p.strip_dirs()
#    print 'cumulative'
#    p.sort_stats('cumulative').print_stats(20)
#    print 'time'
#    p.sort_stats('time').print_stats(20)    

    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp( ibv_resource = tloop )
    app.main()      


if __name__ == '__main__':

#    test_bar2()
#    test_bar4()
    xtest_L_shaped()