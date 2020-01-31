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
# Created on Jun 15, 2010 by: rch

from traits.api import Float, Int

from ibvpy.api import \
    IBVModel, \
    TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
    TLine, BCDof, IBVPSolve as IS, DOTSEval

from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic

from ibvpy.mats.mats1D.mats1D_damage.mats1D_damage import MATS1DDamage

from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
from ibvpy.fets.fets1D.fets1D2l3u import FETS1D2L3U

from numpy import \
    argsort, frompyfunc, linspace, array

from math import exp, fabs

from .rt_nonlocal_averaging import \
    RTNonlocalAvg, QuarticAF

from ibvpy.mesh.fe_grid import FEGrid

import pylab as p

class MATS1DElasticWithFlaw(MATS1DElastic):
    '''Specialized damage model.

    The damage model is driven by a single damage variable omega_0
    at the point x = 0. The material points are damage according
    to the nonlocal distribution function alpha implemnted
    in the get_alpha procedure.

    The implementation reuses the standard MATS1DDamage but replaces
    the state variables at each material point by the single shared
    variable omega_0.
    '''

    flaw_position = Float(0.15)
    flaw_radius = Float(0.05)
    reduction_factor = Float(0.9)

    stiffness = 'secant'

    def get_E(self, sctx=None):

        if sctx:
            X = sctx.fets_eval.get_X_pnt(sctx)
            if fabs(X[0] - self.flaw_position) < self.flaw_radius:
                return self.E * self.reduction_factor
        return self.E

    #-----------------------------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-----------------------------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1, *args, **kw):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        E = self.get_E(sctx)
        D_el = array([[E]])
        sigma = array([E * eps_app_eng[0] ])
        # You print the stress you just computed and the value of the apparent E
        return  sigma, D_el


class BarStrainLocalization(IBVModel):
    '''Model assembling the components for studying the restrained crack localization.
    '''
    shape = Int(10, desc='Number of finite elements',
                   ps_levsls=(10, 40, 4))

    length = Float(1, desc='Length of the simulated region')

    flaw_position = Float(0.5)

    flaw_radius = Float(0.2)

    reduction_factor = Float(0.9)

    avg_radius = Float(0.1)

    def eval(self):

        elem_length = self.length / float(self.shape)
        flaw_radius = self.flaw_radius

        mats = MATS1DElasticWithFlaw(E=10.,
                                     flaw_position=self.flaw_position,
                                     flaw_radius=flaw_radius,
                                     reduction_factor=self.reduction_factor)

        #fets_eval = FETS1D2L( mats_eval = mats )
        fets_eval = FETS1D2L3U(mats_eval=mats)

        domain = FEGrid(coord_max=(self.length, 0., 0.),
                               shape=(self.shape,),
                               fets_eval=fets_eval)

        avg_processor = RTNonlocalAvg(avg_fn=QuarticAF(radius=self.avg_radius,
                                                           correction=True))

        eps_app = RTraceDomainListField(name='Strain' ,
                                         position='int_pnts',
                                         var='eps_app',
                                         warp=False)

        damage = RTraceDomainListField(name='Damage' ,
                                        position='int_pnts',
                                      var='omega',
                                        warp=False)

        disp = RTraceDomainListField(name='Displacement' ,
                                      position='int_pnts',
                                      var='u',
                                      warp=False)

        sig_app = RTraceDomainListField(name='Stress' ,
                                        position='int_pnts',
                                        var='sig_app')

        right_dof = domain[-1, -1].dofs[0, 0, 0]
        rt_fu = RTDofGraph(name='Fi,right over u_right (iteration)' ,
                             var_y='F_int', idx_y=right_dof,
                             var_x='U_k', idx_x=right_dof)

        ts = TS(u_processor=avg_processor,
                dof_resultants=True,
                sdomain=domain,
             # conversion to list (square brackets) is only necessary for slicing of 
             # single dofs, e.g "get_left_dofs()[0,1]"
    #         bcond_list =  [ BCDof(var='u', dof = 0, value = 0.)     ] +  
    #                    [ BCDof(var='u', dof = 2, value = 0.001 ) ]+
    #                    [ )     ],
             bcond_list=[BCDof(var='u', dof=0, value=0.),
    #                        BCDof(var='u', dof = 1, link_dofs = [2], link_coeffs = [0.5],
    #                              value = 0. ),
    #                        BCDof(var='u', dof = 2, link_dofs = [3], link_coeffs = [1.],
    #                              value = 0. ),
                            BCDof(var='u', dof=right_dof, value=0.01,
                                       ) ],
             rtrace_list=[ rt_fu,
                             eps_app,
                             # damage,
                             sig_app,
                             disp,
                    ]
                )

        # Add the time-loop control
        tloop = TLoop(tstepper=ts, KMAX=100, tolerance=1e-5,
                       verbose_iteration=False,
                       tline=TLine(min=0.0, step=1.0, max=1.0))

        U = tloop.eval()

        p.subplot(221)
        rt_fu.refresh()
        rt_fu.trace.plot(p)

        eps = eps_app.subfields[0]
        xdata = eps.vtk_X[:, 0]
        ydata = eps.field_arr[:, 0, 0]
        idata = argsort(xdata)

        p.subplot(222)
        p.plot(xdata[idata], ydata[idata], 'o-')

        disp = disp.subfields[0]
        xdata = disp.vtk_X[:, 0]
        ydata = disp.field_arr[:, 0]
        idata = argsort(xdata)

        p.subplot(223)
        p.plot(xdata[idata], ydata[idata], 'o-')

        sig = sig_app.subfields[0]
        xdata = sig.vtk_X[:, 0]
        ydata = sig.field_arr[:, 0, 0]
        idata = argsort(xdata)

        p.subplot(224)
        p.plot(xdata[idata], ydata[idata], 'o-')

if __name__ == '__main__':

    bsl = BarStrainLocalization(shape=10,
                                 flaw_position=0.5,
                                 reduction_factor=0.99,
                                 flaw_radius=0.1,
                                 avg_radius=0.5)
    bsl.eval()
    p.show()
