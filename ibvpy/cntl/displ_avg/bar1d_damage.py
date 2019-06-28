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

from traits.api import \
    Float, Int, Property, cached_property

from ibvpy.api import \
    IBVModel, \
    TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
    TLine, BCDof, IBVPSolve as IS, DOTSEval

from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic

from ibvpy.mats.mats1D.mats1D_damage.mats1D_damage import MATS1DDamage

from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
from ibvpy.fets.fets1D.fets1D2l3u import FETS1D2L3U

from numpy import \
    argsort, frompyfunc, linspace, array, hstack

from math import exp, fabs

from .rt_nonlocal_averaging import \
    RTNonlocalAvg, QuarticAF, RTUAvg

from ibvpy.mesh.fe_domain import \
    FEDomain

from ibvpy.mesh.fe_refinement_grid import \
    FERefinementGrid

from ibvpy.mesh.fe_grid import FEGrid

import pylab as p

class MATS1DDamageWithFlaw(MATS1DDamage):
    '''Specialized damage model.

    The damage model is driven by a single damaage varialbe omega_0
    at the point x = 0. The material points are damage according
    to the nonlocal distribution function alpha implemnetd
    in the get_alpha procedure.

    The implementation reuses the standard MATS1DDamage but replaces
    the state variables at each material point by the single shared
    variable omega_0.
    '''

    # default parameters of the damage rule
    # onset of damage
    epsilon_0 = 0.1
    # slope of the damage
    epsilon_f = 10.0

    flaw_position = Float(0.15)
    flaw_radius = Float(0.05)
    reduction_factor = Float(0.9)

    stiffness = 'secant'

    def get_epsilon_0(self, sctx=None):

        if sctx:
            X = sctx.fets_eval.get_X_pnt(sctx)
            if fabs(X[0] - self.flaw_position) < self.flaw_radius:
                # print 'reduced epsilon_0 at', X[0], self.epsilon_0 * self.reduction_factor
                return self.epsilon_0 * self.reduction_factor
        return self.epsilon_0

    def _get_omega(self, sctx, kappa):
        epsilon_0 = self.get_epsilon_0(sctx)
        epsilon_f = self.epsilon_f
        if kappa >= epsilon_0 :
            omega = 1. - epsilon_0 / kappa * exp(-1 * (kappa - epsilon_0) / (epsilon_f - epsilon_0))
            return omega
        else:
            return 0.


    def _get_alg_stiffness(self, sctx, eps_app_eng, kappa, omega):
        E = self.E
        D_el = array([E])
        epsilon_0 = self.get_epsilon_0(sctx)
        epsilon_f = self.epsilon_f
        kappa_n = sctx.mats_state_array[0]

        if kappa > epsilon_0 or kappa > kappa_n:
            dodk = epsilon_0 / (kappa ** 2) * exp(-(kappa - epsilon_0) / (epsilon_f - epsilon_0)) + \
                   epsilon_0 / kappa / (epsilon_f - epsilon_0) * exp(-(kappa - epsilon_0) / (epsilon_f - epsilon_0))
            D_alg = (1 - omega) * D_el - D_el * eps_app_eng * dodk
        else:
            D_alg = (1 - omega) * D_el
        return D_alg

class BarStrainLocalization(IBVModel):
    '''Model assembling the components for studying the restrained crack localization.
    '''

    shape = Int(20, desc='Number of finite elements',
                   ps_levsls=(10, 40, 4),
                   input=True)

    length = Float(1, desc='Length of the simulated region',
                    input=True)

    n_steps = Int(5, input=True)

    flaw_position = Float(0.5, input=True)

    flaw_radius = Float(0.1, input=True)

    reduction_factor = Float(0.9, input=True)

    avg_radius = Float(0.4, input=True)

    elastic_fraction = Float(0.8, input=True)

    epsilon_0 = Float(0.1, input=True)

    epsilon_f = Float(10, input=True)

    E = Float(1.0, input=True)

    #---------------------------------------------------------------------------
    # Load scaling adapted to the elastic and inelastic regime
    #---------------------------------------------------------------------------
    final_displ = Property(depends_on='+input')
    @cached_property
    def _get_final_displ(self):
        damage_onset_displ = self.mats.epsilon_0 * self.length
        return damage_onset_displ / self.elastic_fraction

    step_size = Property(depends_on='+input')
    @cached_property
    def _get_step_size(self):
        n_steps = self.n_steps
        return 1.0 / float(n_steps)

    time_function = Property(depends_on='+input')
    @cached_property
    def _get_time_function(self):
        '''Get the time function so that the elastic regime
        is skipped in a single step.
        '''
        step_size = self.step_size

        elastic_value = self.elastic_fraction * 0.98 * self.reduction_factor
        inelastic_value = 1.0 - elastic_value

        def ls(t):
            if t <= step_size:
                return (elastic_value / step_size) * t
            else:
                return elastic_value + (t - step_size) * (inelastic_value) / (1 - step_size)

        return ls

    def plot_time_function(self):
        '''Plot the time function.
        '''
        n_steps = self.n_steps
        mats = self.mats
        step_size = self.step_size

        ls_t = linspace(0, step_size * n_steps, n_steps + 1)
        ls_fn = frompyfunc(self.time_function, 1, 1)
        ls_v = ls_fn(ls_t)

        p.subplot(321)
        p.plot(ls_t, ls_v, 'ro-')

        final_epsilon = self.final_displ / self.length

        kappa = linspace(mats.epsilon_0, final_epsilon, 10)
        omega_fn = frompyfunc(lambda kappa: mats._get_omega(None , kappa), 1, 1)
        omega = omega_fn(kappa)
        kappa_scaled = (step_size + (1 - step_size) * (kappa - mats.epsilon_0) /
                   (final_epsilon - mats.epsilon_0))
        xdata = hstack([array([0.0], dtype=float),
                         kappa_scaled ])
        ydata = hstack([array([0.0], dtype=float),
                         omega])
        p.plot(xdata, ydata, 'g')
        p.xlabel('regular time [-]')
        p.ylabel('scaled time [-]')

    #--------------------------------------------------------------------------------------
    # Time integrator
    #--------------------------------------------------------------------------------------
    mats = Property(depends_on='intput')
    @cached_property
    def _get_mats(self):
        mats = MATS1DDamageWithFlaw(E=self.E,
                                     epsilon_0=self.epsilon_0,
                                     epsilon_f=self.epsilon_f,
                                     flaw_position=self.flaw_position,
                                     flaw_radius=self.flaw_radius,
                                     reduction_factor=self.reduction_factor)
        return mats

    #--------------------------------------------------------------------------------------
    # Space integrator
    #--------------------------------------------------------------------------------------
    fets = Property(depends_on='+input')
    @cached_property
    def _get_fets(self):
        fets_eval = FETS1D2L(mats_eval=self.mats)
        # fets_eval = FETS1D2L3U( mats_eval = self.mats )
        return fets_eval

    fe_domain = Property(depends_on='+input')
    @cached_property
    def _get_fe_domain(self):
        return FEDomain()
    #--------------------------------------------------------------------------------------
    # Mesh integrator
    #--------------------------------------------------------------------------------------
    fe_grid_level = Property(depends_on='+input')
    @cached_property
    def _get_fe_grid_level(self):
        '''Container for subgrids at the refinement level.
        '''
        fe_grid_level = FERefinementGrid(domain=self.fe_domain, fets_eval=self.fets)
        return fe_grid_level

    fe_grid = Property(depends_on='+input')
    @cached_property
    def _get_fe_grid(self):

        elem_length = self.length / float(self.shape)

        fe_grid = FEGrid(coord_max=(self.length,),
                          shape=(self.shape,),
                          level=self.fe_grid_level,
                          fets_eval=self.fets)
        return fe_grid

    #--------------------------------------------------------------------------------------
    # Tracers
    #--------------------------------------------------------------------------------------
    rt_list = Property(depends_on='+input')
    @cached_property
    def _get_rt_list(self):
        '''Prepare the list of tracers
        '''
        right_dof = self.fe_grid[-1, -1].dofs[0, 0, 0]

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

        rt_fu = RTDofGraph(name='Fi,right over u_right (iteration)' ,
                             var_y='F_int', idx_y=right_dof,
                             var_x='U_k', idx_x=right_dof)

        return [ rt_fu, eps_app, damage, sig_app, disp ]

    def plot_tracers(self):

        rt_fu, eps_app, damage, sig_app, disp = self.rt_list

        p.subplot(323)

        p.xlabel('control displacement [m]')
        p.ylabel('load [N]')

        rt_fu.refresh()
        rt_fu.trace.plot(p, 'o-')

        disp = disp.subfields[0]
        xdata = disp.vtk_X[:, 0]
        ydata = disp.field_arr[:, 0]
        idata = argsort(xdata)

        p.subplot(325)
        p.plot(xdata[idata], ydata[idata], 'o-')
        p.xlabel('bar axis [m]')
        p.ylabel('displacement [m]')

        eps = eps_app.subfields[0]
        xdata = eps.vtk_X[:, 0]
        ydata = eps.field_arr[:, 0, 0]
        idata = argsort(xdata)

        p.subplot(322)
        p.plot(xdata[idata], ydata[idata], 'o-')
        p.ylim(ymin=0)
        p.xlabel('bar axis [m]')
        p.ylabel('strain [-]')

        damage = damage.subfields[0]
        xdata = damage.vtk_X[:, 0]
        ydata = damage.field_arr[:]
        idata = argsort(xdata)

        p.subplot(324)
        p.plot(xdata[idata], ydata[idata], 'o-')
        p.ylim(ymax=1.0, ymin=0)
        p.xlabel('bar axis [m]')
        p.ylabel('damage [-]')

        sig = sig_app.subfields[0]
        xdata = sig.vtk_X[:, 0]
        ydata = sig.field_arr[:, 0, 0]
        idata = argsort(xdata)
        ymax = max(ydata)

        p.subplot(326)
        p.plot(xdata[idata], ydata[idata], 'o-')
        p.ylim(ymin=0, ymax=1.2 * ymax)
        p.xlabel('bar axis [m]')
        p.ylabel('stress [N]')

    bc_list = Property(depends_on='+input')
    @cached_property
    def _get_bc_list(self):
        '''List of boundary concditions
        '''
        right_dof = self.fe_grid[-1, -1].dofs[0, 0, 0]

        bcond_list = [ BCDof(var='u', dof=0, value=0.),
                      BCDof(var='u', dof=right_dof, value=self.final_displ,
                             time_function=self.time_function
                             ) ]
        return bcond_list

    def eval(self):
        '''Run the time loop.
        '''
        #
        avg_processor = None
        if self.avg_radius > 0.0:
            avg_processor = RTNonlocalAvg(sd=self.fe_domain,
                                           avg_fn=QuarticAF(radius=self.avg_radius,
                                                               correction=True))

        ts = TS(u_processor=avg_processor,
                 dof_resultants=True,
                 sdomain=self.fe_domain,
                 bcond_list=self.bc_list,
                 rtrace_list=self.rt_list
                )

        # Add the time-loop control
        tloop = TLoop(tstepper=ts, KMAX=300, tolerance=1e-8,
                       debug=False,
                       verbose_iteration=False,
                       verbose_time=False,
                       tline=TLine(min=0.0, step=self.step_size, max=1.0))

        tloop.eval()

        tloop.accept_time_step()

        self.plot_time_function()
        self.plot_tracers()


if __name__ == '__main__':

    do = 'avg_test'

    if do == 'avg_test':
        avg_radius_list = [ 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ]
        # avg_radius_list = [ 0.0, 0.05, 0.1, 0.3, 0.7 ]
        # avg_radius_list = [ 0.1, 0.6 ]
        # avg_radius_list = [0.1]
        legend_list = [ 'radius = %.2f' % radius for radius in avg_radius_list ]

        shape = 51
        n_steps = 50
        length = 1
        flaw_radius = length / float(shape) / 2.0

        bsl = BarStrainLocalization(shape=shape,
                                     E=100.0,
                                     epsilon_0=0.1,
                                     epsilon_f=1.0,
                                     n_steps=n_steps,
                                     length=length,
                                     elastic_fraction=0.7,
                                     reduction_factor=0.9,
                                     flaw_position=0.5,
                                     flaw_radius=flaw_radius)

        for avg_radius in avg_radius_list:
            bsl.avg_radius = avg_radius
            bsl.eval()

        p.legend(legend_list)
        p.show()
    elif do == 'alg_test':

        shape = 3
        n_steps = 2
        length = 1

        flaw_radius = length / float(shape) / 2.0

        bsl = BarStrainLocalization(shape=shape,
                                     E=100.0,
                                     n_steps=n_steps,
                                     length=length,
                                     elastic_fraction=0.9,
                                     reduction_factor=0.9,
                                     flaw_position=0.5,
                                     flaw_radius=flaw_radius,
                                     avg_radius=0.6,
                                     epsilon_0=0.2,
                                     epsilon_f=1.0
                                     )

        bsl.eval()
        p.show()
