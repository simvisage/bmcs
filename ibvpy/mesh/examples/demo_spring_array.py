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
# Created on Oct 26, 2010 by: rch

from ibvpy.api import\
     BCSlice, TStepper as TS, TLoop, TLine, RTDofGraph
from ibvpy.rtrace.rt_domain_list_field import RTraceDomainListField
from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.mesh.fe_refinement_grid import FERefinementGrid
from ibvpy.mesh.fe_domain import FEDomain

from ibvpy.mesh.fe_spring_array import FESpringArray
from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
from ibvpy.mats.mats1D import MATS1DElastic

if __name__ == '__main__':

    fets_eval = FETS1D2L( mats_eval = MATS1DElastic( E = 1 ) )

    # Discretization
    fe_domain = FEDomain()

    fe_patch_left = FERefinementGrid( name = 'left',
                                     fets_eval = fets_eval,
                                     domain = fe_domain )

    fe_grid_left = FEGrid( level = fe_patch_left,
                      coord_min = ( 0., ),
                      coord_max = ( 1., ),
                      shape = ( 1, ),
                      fets_eval = fets_eval )

    fe_patch_right = FERefinementGrid( name = 'refinement grid',
                                       fets_eval = fets_eval,
                                       domain = fe_domain )

    fe_grid_right = FEGrid( level = fe_patch_right,
                            coord_min = ( 2., ),
                            coord_max = ( 3., ),
                            shape = ( 1, ),
                            fets_eval = fets_eval )

    dofs_left = fe_grid_left[-1, -1].dofs[0, :, 0]
    dofs_right = fe_grid_right[0, 0].dofs[0, :, 0]
    spring_arr = FESpringArray( domain = fe_domain,
                                dofs_1 = dofs_left,
                                dofs_2 = dofs_right, k_value = 1.0 )

    ts = TS( dof_resultants = True,
             sdomain = fe_domain,
             bcond_list = [BCSlice( var = 'u', value = 1., dims = [0],
                                       slice = fe_grid_right[ -1, -1 ] ),
                           BCSlice( var = 'u', value = 0., dims = [0],
                                       slice = fe_grid_left[ 0, 0 ] )
                                       ],
             rtrace_list = [  ]
                )

    # Add the time-loop control
    tloop = TLoop( tstepper = ts,
                   tline = TLine( min = 0.0, step = 1, max = 1.0 ) )

    u = tloop.eval()
    print(u)

    import pylab as p
    spring_arr.plot_spring_forces( u, p )
    p.show()

#    from ibvpy.plugins.ibvpy_app import IBVPyApp
#    ibvpy_app = IBVPyApp( ibv_resource = tloop )
#    ibvpy_app.main()

