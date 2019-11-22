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
# Created on Jan 21, 2011 by: rch


from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
from ibvpy.api import FEDomain, FEGrid, FERefinementGrid, TStepper as TS, TLoop, \
    BCSlice, RTraceDomainListField
from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
from ibvpy.fets.fets1D.fets1D2l import FETS1D2L

if __name__ == '__main__':

    if True:
        fets_eval_4u = FETS2D4Q( mats_eval = MATS2DElastic() )
        fe_grid = FEGrid( name = 'fe_grid1', coord_max = ( 4., 4. ),
                                   shape = ( 4, 4 ),
                                   fets_eval = fets_eval_4u )

        interior_elems = fe_grid[ 1:3, 1:3, :, : ].elems
        interior_bc = fe_grid[ 1, 1, 1:, 1: ]

        bcond_list = [BCSlice( var = 'u', dims = [0, 1], slice = fe_grid[ :, 0, :, 0 ], value = 0.0 ),
                      BCSlice( var = 'u', dims = [0, 1], slice = interior_bc,
                               link_slice = fe_grid[1, 0, 0, 0], link_coeffs = [0], value = 0.0 ),
                      BCSlice( var = 'f', dims = [1], slice = fe_grid[ 0, -1, :, -1 ], value = 1.0 ) ]

    else:
        fets_eval_1d = FETS1D2L( mats_eval = MATS1DElastic() )
        fe_grid = FEGrid( name = 'fe_grid1', coord_max = ( 4., ),
                                   shape = ( 4, ),
                                   fets_eval = fets_eval_1d )

        interior_elems = fe_grid[ 1:3, : ].elems
        interior_bc = fe_grid[ 1:2, 1: ]

        bcond_list = [BCSlice( var = 'u', dims = [0], slice = fe_grid[ 0, 0 ], value = 0.0 ),
                      BCSlice( var = 'u', dims = [0], slice = interior_bc,
                               link_slice = fe_grid[0, 0], link_coeffs = [0], value = 0.0 ),
                      BCSlice( var = 'u', dims = [0], slice = fe_grid[ -1, -1 ], value = 1.0 ) ]

    fe_grid.inactive_elems = list( interior_elems )
    print('elems', interior_bc.elems)
    print('dofs', interior_bc.dofs)
    print('nodes', interior_bc.dof_nodes)

    rtrace_list = [RTraceDomainListField( name = 'sig_app' ,
                                      var = 'sig_app',
                                      record_on = 'update', ) ]

    ts = TS( sdomain = fe_grid,
             bcond_list = bcond_list,
             rtrace_list = rtrace_list,
             )
    tloop = TLoop( tstepper = ts,
                    )

    u = tloop.eval()

#    from ibvpy.plugins.ibvpy_app import IBVPyApp
#    ibvpy_app = IBVPyApp( ibv_resource = tloop )
#    ibvpy_app.main()
