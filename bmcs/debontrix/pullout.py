'''
Created on Mar 4, 2017

@author: rch

Run #1
   - add plot2d adaptors to boundary conditions
   - fix the natural boundary conditions
   - add BMCSTreeNode base class
   - add the calculation button, add the proces button
   - add splash screen
   - entry point

Run #2
   - efficiency of the boundary conditions - index notation
   - sparse-solver warning - chcek
   - coupling of domains
   - coupling with reinforcement - non-conforming
   
Run #3
    - domain with two layers - two dimensional
    - domain with 3d sliding tensor - what material 
      would it correspond to?
      
Run #4
    - Cohesive crack
    - Crack localization - strain softening
    - Explain dissipation - what happens if stiffness changes
    - What happens upon unloading
'''
import numpy as np

if __name__ == '__main__':
    from ibvpy.api import \
        TStepper as TS, RTraceGraph, RTraceDomainListField, TLoop, \
        TLine, BCSlice
    from ibvpy.dots.dots_grid_eval import DOTSGridEval
    from debontrix import FETS1D2L

    A_fiber = 1.
    E_fiber = 1.
    stiffness_fiber = E_fiber * A_fiber

    d = 2 * np.sqrt(np.pi)
    tau_max = 0.1 * d * np.pi
    G = 100
    f_max = 0.2

    fets_eval = FETS1D2L()
    tse = DOTSGridEval(
        n_E=8,
        L_x=1.0,
        G=1.0,
        fets_eval=fets_eval)

    ts = TS(dof_resultants=True,
            tse=tse,
            sdomain=tse.sdomain,
            # conversion to list (square brackets) is only necessary for slicing of
            # single dofs, e.g "get_left_dofs()[0,1]"
            bcond_list=[
                BCSlice(var='u', value=0., dims=[0],
                        slice=tse.sdomain[0, 0]),
                BCSlice(var='u', value=0.1, dims=[1],
                        slice=tse.sdomain[-1, -1])
            ],
            #             rtrace_list=[RTraceGraph(name='Fi,right over u_right (iteration)',
            #                                      var_y='F_int', idx_y=end_dof,
            #                                      var_x='U_k', idx_x=end_dof),
            #                          RTraceDomainListField(name='slip',
            #                                                var='slip', idx=0),
            #                          RTraceDomainListField(name='eps1',
            #                                                var='eps1', idx=0),
            #                          RTraceDomainListField(name='eps2',
            #                                                var='eps2', idx=0),
            #                          RTraceDomainListField(name='shear_flow',
            #                                                var='shear_flow', idx=0),
            #                          RTraceDomainListField(name='sig1',
            #                                                var='sig1', idx=0),
            #                          RTraceDomainListField(name='sig2',
            #                                                var='sig2', idx=0),
            #                          RTraceDomainListField(name='Displacement',
            #                                                var='u', idx=0)
            #                          ]
            )
    # Add the time-loop control
    tloop = TLoop(tstepper=ts, KMAX=30, debug=False,
                  tline=TLine(min=0.0, step=0.1, max=1.0))

    print tloop.eval()
