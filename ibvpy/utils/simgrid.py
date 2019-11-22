#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Aug 13, 2009 by: rch

from ibvpy.api import \
    TStepper as TS, RTDofGraph, RTraceDomainListField, RTraceDomainListInteg, TLoop, \
    TLine, BCDofGroup, IBVPSolve as IS, DOTSEval, BCSlice

from ibvpy.mesh.fe_grid import FEGrid
from numpy import column_stack

def simgrid( fets_eval, 
             cube_size,
             shape,
             support_slices,
             support_dirs,
             loading_slice,
             load_dir = 0,
             load_max = 0.01,
             n_load_steps = 1,
             vars = [],
             ivars = [],
             var_type = 'u',
              ):
    '''Construct an idealization and run simulation with primary variable 
    fixed on support slices and with unit load on loading slices applied in load dir.
    Return the solution vector and the fields specified in vars.
    '''
    # Discretization
    domain = FEGrid( coord_max = cube_size, 
                     shape   = shape,
                     fets_eval = fets_eval )

    u_max = load_max

    support_bcond = [ BCSlice( var = 'u', value = 0, 
                               dims = support_dir, 
                               slice = domain[support_slice] )
                      for support_slice, support_dir in zip( support_slices, support_dirs ) ]
    load_bcond    = [ BCSlice( var = var_type, value = u_max, 
                               dims = [load_dir], 
                               slice = domain[loading_slice] ) ]

    bcond = support_bcond + load_bcond 

    loading_dofs = domain[loading_slice].dofs[:,:,load_dir].flatten()

    graphs = [ RTDofGraph( name = 'Force in one node / Displ.',
                              var_y = 'F_int', idx_x = loading_dof,
                              var_x = 'U_k'  , idx_y = loading_dof )
                 for loading_dof in loading_dofs ]

    rtrace_list = [ RTraceDomainListField( name = var ,
                                           var = var,
                                           warp = True, 
                                           #position = 'int_pnts',
                                           record_on = 'update' )
                    for var in vars
                    ]
    irtrace_list = [ RTraceDomainListInteg( name = 'Integ(' + var + ')' ,
                                            var = var,
                                            record_on = 'update' )
                    for var in ivars ]
    
    ts = TS(
            sdomain = domain,
            bcond_list = bcond,
            rtrace_list = graphs + rtrace_list + irtrace_list
            )

    load_step = 1. / float( n_load_steps )
    # Add the time-loop control
    tloop = TLoop( tstepper = ts, KMAX = 15, RESETMAX = 0, tolerance = 1e-5,
                        tline  = TLine( min = 0.0,  step = load_step, max = 1.0 ))

    u = tloop.eval()
    
    fields = [ rtrace.subfields[0].field_arr for rtrace in rtrace_list ]
    integs = [ rtrace.integ_val for rtrace in irtrace_list ]
    for graph in graphs:
        graph.redraw()
    traces = [ graph.trace for graph in graphs ]
    xydata = ( traces[0].xdata, column_stack( [trace.ydata for trace in traces ] ) )

    return tloop, u, fields, integs, xydata

if __name__ == '__main__':

    from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import MATS3DElastic
    from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
    
    from ibvpy.fets.fets3D.fets3D8h import FETS3D8H
    from ibvpy.fets.fets2D5.fets2D58h import FETS2D58H

    fets_eval_3D = FETS3D8H(mats_eval  = MATS3DElastic(E = 34000, nu = 0.25))        

    support_slices = [
                      [ (0   ,slice(None),slice(None),0   ,slice(None),slice(None)), # yz plane  0
                        (0   ,0   ,slice(None),0   ,0   ,slice(None)), #  z-axis   1
                        (0   ,0   ,   0,0   ,0   ,0   )  #  origin   2
                      ],
                      [ 
                        (0   ,0   ,0   ,0   ,0   ,0   ), #  origin   0
                        (slice(None),0   ,slice(None),slice(None),0   ,slice(None)), # xz plane  1
                        (slice(None),0   ,0   ,slice(None),0   ,0   ), #  y-axis   2
                      ],
                      [ 
                        (0   ,slice(None),0   ,0   ,slice(None),0   ), #  x-axis   0
                        (0   ,0   ,0   ,0   ,0   ,0   ), #  origin   1
                        (slice(None),slice(None),0   ,slice(None),slice(None),0   ), # xy plane  2
                      ], 
                      ]
    support_dirs = [[0],[1],[2]]
    
    loading_slices = [ 
                      (-1  ,slice(None),slice(None),-1  ,slice(None),slice(None)),  # loading in x dir
                      (slice(None),-1  ,slice(None),slice(None),-1  ,slice(None)),  # loading in y dir
                      (slice(None),slice(None),-1  ,slice(None),slice(None),-1  )   # loading in z dir
                    ]

    tl, u1, fields, integs, g = simgrid( fets_eval_3D, (3,3,3), (1,1,1), 
                              support_slices[0], support_dirs,
                              loading_slices[0], 0, 0.01, 1, [] )
    
    tl, u2, fields, integs, g = simgrid( fets_eval_3D, (3,3,3), (1,1,1), 
                              support_slices[1], support_dirs,
                              loading_slices[1], 1, 0.01, 1, ['u'] )
    
    print('u1')
    for idx, u in enumerate( u1 ):
        print('[', idx, ']', u)
    
    print('u2')
    for idx, u in enumerate( u2 ):
        print('[', idx, ']', u)
    
    
        