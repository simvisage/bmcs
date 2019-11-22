'''
Created on Jul 20, 2010

@author: jakub
'''
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
# Created on Jun 23, 2010 by: rch

from ibvpy.api import \
    TStepper as TS, RTDofGraph, TLoop, \
    TLine, BCDof, IBVPSolve as IS, DOTSEval, FEDomain, FERefinementGrid, \
    FEGrid, BCSlice, RTraceDomainListField


#from apps.scratch.jakub.mlab.mlab_trace import RTraceDomainListField
from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.mats.mats2D.mats2D_sdamage.strain_norm2d import Euclidean, Mazars, Rankine
from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
from ibvpy.fets.fets2D.fets2D4q9u import  FETS2D4Q9U
from ibvpy.fets.fets2D.fets2D4q8u import FETS2D4Q8U
from numpy import array, cos, sin, pi, sqrt, deg2rad, arctan
from mathkit.mfn import MFnLineArray

from .rt_nonlocal_averaging import \
    RTNonlocalAvg, QuarticAF

def app():
    avg_radius = 0.03

    md = MATS2DScalarDamage(E=20.0e3,
                            nu=0.2,
                            epsilon_0=1.0e-4,
                            epsilon_f=8.0e-4,
                            #epsilon_f = 12.0e-4, #test doubling the e_f
                            stress_state="plane_strain",
                            stiffness="secant",
                            #stiffness  = "algorithmic",
                            strain_norm=Rankine())
    me = MATS2DElastic(E=20.0e3,
                       nu=0.2,
                       stress_state="plane_strain")

    fets_eval = FETS2D4Q(mats_eval=md)#, ngp_r = 3, ngp_s = 3)                                               

    n_el_x = 40
    # Discretization
    fe_grid = FEGrid(coord_max=(.6, .2, 0.),
                      shape=(n_el_x, n_el_x / 3),
                      fets_eval=fets_eval)

    mf = MFnLineArray(xdata=array([0, 1, 2, 3, 4 ]),
                       ydata=array([0, 2., 2.5, 3., 3.2 ]))

    #averaging function
    avg_processor = RTNonlocalAvg(avg_fn=QuarticAF(radius=avg_radius,
                                                       correction=True))

    ts = TS(sdomain=fe_grid,
             u_processor=avg_processor,
             bcond_list=[
                        # constraint for all left dofs in y-direction:
                        BCSlice(var='u', slice=fe_grid[0, 0, 0, 0], dims=[0, 1], value=0.),
                        BCSlice(var='u', slice=fe_grid[-1, 0, -1, 0], dims=[1], value=0.),
                        BCSlice(var='u', slice=fe_grid[n_el_x / 2, -1, 0, -1], dims=[1],
                                time_function=mf.get_value,
                                value= -2.0e-5),
                        ],
             rtrace_list=[
                            RTDofGraph(name='Fi,right over u_right (iteration)' ,
                                      var_y='F_int', idx_y=right_dof,
                                      var_x='U_k', idx_x=right_dof,
                                      record_on='update'),
                            RTraceDomainListField(name='Strain' ,
                                           var='eps_app', idx=0,
                                           record_on='update'),
                            RTraceDomainListField(name='Displacement' ,
                                           var='u', idx=1,
                                           record_on='update',
                                           warp=True),
                            RTraceDomainListField(name='Damage' ,
                                           var='omega', idx=0,
                                           record_on='update',
                                           warp=True),
    #                         RTraceDomainField(name = 'Stress' ,
    #                                        var = 'sig', idx = 0,
    #                                        record_on = 'update'),
    #                        RTraceDomainField(name = 'N0' ,
    #                                       var = 'N_mtx', idx = 0,
    #                                       record_on = 'update')
                        ]
            )

    # Add the time-loop control
    #
    tl = TLoop(tstepper=ts,
                tolerance=5.0e-5,
                KMAX=50,
                tline=TLine(min=0.0, step=1., max=10.0))
    tl.eval()
    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp(ibv_resource=ts)
    ibvpy_app.main()

if __name__ == '__main__':
    app()
