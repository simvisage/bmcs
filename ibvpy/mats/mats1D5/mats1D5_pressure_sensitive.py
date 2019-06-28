#-------------------------------------------------------------------------
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
# Created on Sep 8, 2009 by: rch

import math

from ibvpy.mats.mats1D5.mats1D5_eval import MATS1D5Eval
from ibvpy.mats.mats_eval import IMATSEval
from mathkit.numpy.numpy_func import Heaviside
from traits.api import \
    Instance, Property, cached_property, List, \
    Callable, String, Int, HasTraits, Float
from traitsui.api import \
    View, Item

import numpy as np


class MATS1D5PressureSensitive(MATS1D5Eval):
    r'''Bond model for two phases interacting over an interface with zero thickness.
The frictional resistance is governed by the current level of pressure 
using the cohesion :math:`c` and frictional angle :math`\phi`. 

Basic formulas

The threshold level of frictional stress is prescribed by the Mohr-Coulomb rule:

.. math::
    \tau_\mathrm{fr}(\sigma) = \left( c + \sigma \; \tan{(\phi)} \right) 
     H\left( \sigma -  \frac{c}{\tan{(\phi)}}\right)
    :label: eq_MATS1D5PressureSensitive

The deformation variables include interface opening :math:`w` and sliding :math:`s`
related to the stress variables using the equations

.. math::
    \tau = G_s \cdot ( s - s^p )

.. math::
    \sigma = \left( G_w^{(+)} H(w) + G_w^{(-)} H(-w) \right) \cdot w

where 
 - :math:`G_s` represents elastic shear stiffness, 
 - :math:`G_w^{(+)}` is the adhesive stiffness for crack opening and 
 - :math:`G_w^{(-)}` is the penetration (or penalty) stiffness to be set large.

The elastic domain is given by the yield condition

.. math::
    f := \left| \tau \right| - \tau_\mathrm{fr}(\sigma) \leq 0

Thus, the Kuhn-Tucker condition must hold:

.. math::
    \dot{\gamma} \geq 0, \; f( \sigma, \tau ) \cdot \gamma \leq 0, \;
    \dot{\gamma} f(\sigma, \tau) = 0

where :math:`\gamma` represents the plastic multiplier.     
The flow rule for yielding slip is provided as

.. math::
    \dot{s}_p = \dot{\gamma} \; \mathrm{sign}(\tau)

and the consistency condition (staying on the yield surface upon platic loading)

.. math::
    \dot{\gamma} \dot{f}(\sigma, \tau) = 0

Discrete form

Using the midpoint rule 

Given the displacement increment 

.. math::
    s_{n+1} = s_{n} + \Delta s_{n}, \; 
    w_{n+1} = w_{n} + \Delta w_{n}, \;
    :label: eq_increment

the elastic trial stresses are calculated as

.. math::
    \tau_{n+1}^{\mathrm{trial}} = G_s ( s_{n+1} - s_n^p ), \;
    \sigma_{n+1} = G_w ( w_{n+1} )
    :label: eq_constitutive_law

and the frictional threshold for the current transverse stresses

.. math::
    \tau^\mathrm{fr}_{n+1}(\sigma_{n+1}) = 
    \left( c + \sigma_{n+1} \; \tan{(\phi)} \right) 
     H\left( \sigma_{n+1} -  \frac{c}{\tan{(\phi)}}\right)
    :label: eq_mohr_coulomb

The trial consistency condition is obtained as

.. math::
    f_{n+1}^\mathrm{trial} = \left| \tau_{n+1}^\mathrm{trial} \right|
    - \tau^\mathrm{fr}_{n+1}(\sigma_{n+1})
    :label: eq_discr_consistency

If :math:`f^\mathrm{trial}_{n+1} \leq 0` then step is elastic.

If :math:`f^\mathrm{trial}_{n+1} > 0 \Leftrightarrow f(\tau_{n+1}, \sigma_{n+1} ) = 0`
The objective is to identify :math:`s^{p}_{n+1}, \Delta \gamma` satisfying this condition
and the condition of consistency. To accomplish this task we first note that

.. math::
    \tau_{n+1}= G_s( s_{n+1} - s^p_{n+1} ) =G_s( s_{n+1} - s^p_{n}) - G_s( s^p_{n+1} - s^p_n )

Using Eq.:eq:`eq_midpoint` the shear stress can be related to trial state and the plastic 
(return mapping) multiplier :math:`\Delta \gamma` as 

.. math::
    \tau_{n+1} = \tau^{\mathrm{trial}}_{n+1} - G_s( s^p_{n+1} - s^p_n )
    = \tau^{\mathrm{trial}}_{n+1} - G_s \Delta \gamma \, \mathrm{sign}( \tau_{n+1} ).

The other two incremental equation deliver the plastic slip and consistency condition.

.. math::
    s^{p}_{n+1} = s^p_n + \Delta \gamma \, \mathrm{sign}( \tau_{n+1} )
    :label: eq_midpoint

.. math::
    f_{n+1} = \left| \tau_{n+1} \right| - \tau^{\mathrm{fr}}(\sigma_{n+1}) = 0
    :label: eq_discr_consistency2

It can be shown that the direction of mapping given by :math:`\mathrm{sign}(\tau_{n+1})`
is consistent with the trial state, i,e. :math:`\mathrm{sign}(\tau^{\mathrm{trial}}_{n+1})`
and

.. math::
    \left| \tau_{n+1} \right| + \Delta \gamma G_s = \left| \tau^{\mathrm{trial}}_{n+1} \right|

Finally, due to :math:`\Delta \gamma > 0` the discrete consistency condition 
(:eq:`eq_discr_consistency`) can be further expanded as

.. math::
    f_{n+1} = \left| \tau^{\mathrm{trial}}_{n+1} \right| - 
    G_s \Delta \gamma- \tau^{\mathrm{fr}}( \sigma_{n+1} )

Hence

.. math::
    f_{n+1} = 0 \implies \Delta \gamma = \frac{ f^{\mathrm{trial}}_{n+1}}{ G_s } > 0

and

.. math::
    \tau_{n+1} &= \tau^{\mathrm{trial}}_{n+1} - 
    \Delta \gamma G_s \, \mathrm{sign}( \tau_{n+1}^{\mathrm{trial}} ) \\
    s^p_{n+1} &= s^p_n + 
    \Delta \gamma \, \mathrm{sign}( \tau_{n+1}^{\mathrm{trial}} ) \\ 

    '''

    # implements(IMATSEval)

    G_s = Float(1.0, input=True, enter_set=False,
                label='cohesion')

    G_w_open = Float(1.0, input=True, enter_set=False,
                     label='cohesion')

    G_w_close = Float(1e+6, input=True, enter_set=False,
                      label='cohesion')

    c = Float(0.0, input=True, enter_set=False,
              label='cohesion')

    phi = Float(0.1, input=True, enter_set=False,
                label='friction angle')

    #-------------------------------------------------------------------------
    # Submodels constituting the interface behavior
    #-------------------------------------------------------------------------

    traits_view = View(Item('c@'),
                       Item('phi@'),
                       Item('G_open@'),
                       Item('G_slip@'),
                       Item('G_penalty@'),
                       resizable=True,
                       scrollable=True,
                       width=0.8,
                       height=0.9,
                       buttons=['OK', 'Cancel'])

    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------

    state_var_shapes = {'slip_p': ()}

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, u_r, tn1, slip_p):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''

        s_n = u_r[..., 0]
        w = u_r[..., 1]
        s_p_n = slip_p

        f_separate = (Heaviside(w) * self.G_open + 
                      Heaviside(-w) * self.G_penalty) * w

        f_r = np.zeros_like(u_r)
        tau_trial = self.G_slip * (s_n - s_p_n)
        f_trial = abs(tau_trial) - (f_separate + math.tan(self.phi))

        f_I = np.where(f_trial > 0.0)
        f_r[..., 0] = tau_trial

        D_n1[0, 0] = E
        d_gamma = f_trial / (self.E + self.K_bar + self.H_bar)
        sig_n1[0] = sigma_trial - d_gamma * self.E * sign(xi_trial)
        D_n1[0, 0] = (self.E * (self.K_bar + self.H_bar)) / \
            (self.E + self.K_bar + self.H_bar)

        sig_n1 = zeros((1,), dtype='float_')
        D_n1 = zeros((1, 1), dtype='float_')

        D_mtx = zeros(
            (eps_app_eng.shape[0], eps_app_eng.shape[0]), dtype='float_')

        return sig_app_eng, D_mtx

    def get_sig1(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return sig_eng[0:1]

    def get_sig2(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return sig_eng[3:]

    def get_shear_flow(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return sig_eng[1:2]

    def get_cohesive_stress(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return sig_eng[2:3]

    rte_dict = Property

    def _get_rte_dict(self):

        rte_dict = {}
        ix_maps = [0, 1, 2, 3]
        for name, mats, ix_map, size, offset in \
            zip(self._mats_names, self._mats_list, ix_maps, self._state_sizes,
                self._state_offsets):
            for key, v_eval in list(mats.rte_dict.items()):

                __call_v_eval = RTE1D5Bond(v_eval=v_eval,
                                           name=name + '_' + key,
                                           size=size,
                                           offset=offset,
                                           ix_map=ix_map)

                rte_dict[name + '_' + key] = __call_v_eval

        # sigma is also achievable through phase1_sig_app and phase_2_sig_app
        extra_rte_dict = {'sig1': self.get_sig1,
                          'sig2': self.get_sig2,
                          'shear_flow': self.get_shear_flow,
                          'cohesive_stress': self.get_cohesive_stress,
                          }
        rte_dict.update(extra_rte_dict)
        return rte_dict
    #-------------------------------------------------------------------------
    # Methods required by the mats_explore tool
    #-------------------------------------------------------------------------

    def new_cntl_var(self):
        return zeros(4, 'float_')

    def new_resp_var(self):
        return zeros(4, 'float_')


class RTE1D5Bond(HasTraits):

    v_eval = Callable
    name = String
    size = Int
    offset = Int
    ix_map = Int

    def __call__(self, sctx, u, *args, **kw):
        u_x = array([u[self.ix_map]], dtype='float')
        # save the spatial context
        mats_state_array = sctx.mats_state_array
        sctx.mats_state_array = mats_state_array[self.offset: self.offset + self.size]
        result = self.v_eval(sctx, u_x, *args, **kw)
        # put the spatial context back
        sctx.mats_state_array = mats_state_array

        return result


def sp_derive():

    import sympy as sp

    vars = 'G_s, s_n, s_p_n, w_n, dw_n, ds_n, G_s, G_w, c, phi'

    syms = sp.symbols(vars)

    for var, sym in zip(vars.split(','), syms):
        globals()[var.strip()] = sym

    s_n1 = s_n + ds_n
    w_n1 = w_n + dw_n

    tau_trial = G_s * (s_n1 - s_p_n)

    print('diff', sp.diff(tau_trial, ds_n))

    print(tau_trial)

    sig_n1 = G_w * w_n1

    print(sig_n1)

    tau_fr = (c + sig_n1 * sp.tan(phi)) * \
        sp.Heaviside(sig_n1 - c / sp.tan(phi))

    print(tau_fr)

    d_tau_fr = sp.diff(tau_fr, dw_n)

    print(d_tau_fr)

    f_trial = sp.abs(tau_trial) - tau_fr

    print(f_trial)

    d_gamma = f_trial / G_s

    print('d_gamma')
    sp.pretty_print(d_gamma)

    print('d_gamma_s')
    sp.pretty_print(sp.diff(d_gamma, ds_n))

    print('tau_n1')
    tau_n1 = sp.simplify(tau_trial - d_gamma * G_s * sp.sign(tau_trial))
    sp.pretty_print(tau_n1)

    print('dtau_n1_w')
    dtau_n1_w = sp.diff(tau_n1, dw_n)
    sp.pretty_print(dtau_n1_w)

    print('dtau_n1_s')
    dtau_n1_s = sp.diff(d_gamma * sp.sign(tau_trial), ds_n)
    print(dtau_n1_s)

    s_p_n1 = s_p_n + d_gamma * sp.sign(tau_trial)

    print(s_p_n1)


if __name__ == '__main__':

    sp_derive()
