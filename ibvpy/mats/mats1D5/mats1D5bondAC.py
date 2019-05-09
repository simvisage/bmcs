'''
Created on Mar 29, 2009

@author: jakub
'''

from math import pi as Pi, cos, sin, exp, sqrt as scalar_sqrt

from ibvpy.api import RTrace, RTDofGraph, RTraceArraySnapshot
from ibvpy.core.tstepper import \
    TStepper as TS
from ibvpy.mats.mats_eval import IMATSEval, MATSEval
from mathkit.mfn import MFnLineArray
from numpy import \
    array, ones, zeros, outer, inner, transpose, dot, frompyfunc, \
    fabs, sqrt, linspace, vdot, identity, tensordot, \
    sin as nsin, meshgrid, float_, ix_, \
    vstack, hstack, sqrt as arr_sqrt
from scipy.linalg import eig, inv
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, \
    Instance, Int, Trait, Range, HasTraits, on_trait_change, Event, \
    Dict, Property, cached_property, Delegate
from traitsui.api import \
    Item, View, HSplit, VSplit, VGroup, Group, Spring


from .mats1D5bond import MATS1D5Bond


# Chaco imports
# from dacwt import DAC
#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS1D5BondAC(MATS1D5Bond):
    '''
    Adhesive Cohesive Bond Model
    '''

    s_cr = Float(1.,  # 34e+3,
                 label="s_cr",
                 desc="Critical Slip",
                 auto_set=False)
    tau_max = Float(1.,  # 34e+3,
                    label="T_max",
                    desc="maximal shear stress",
                    auto_set=False)
    tau_fr = Float(1.,  # 34e+3,
                   label="T_fr",
                   desc="Frictional shear stress",
                   auto_set=False)

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = Event

    #-------------------------------------------------------------------------
    # View specification
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # Private initialization methods
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------

    def setup(self, sctx):
        '''
        Intialize state variables.
        '''
        sctx.mats_state_array[:] = 0

    def get_state_array_size(self):
        '''
        Return number of number to be stored in state array
        @param sctx:spatial context
        '''
        return 2  # TODO: works just with linear element

    def new_cntl_var(self):
        return zeros(4, float_)  # TODO: adapt for 4-5..

    def new_resp_var(self):
        return zeros(4, float_)  # TODO: adapt for 4-5..

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        if sctx.update_state_on:
            omega_l = omega_r = 0
            if eps_app_eng[0] > self.s_cr:
                omega_l = 1
            if eps_app_eng[1] > self.s_cr:
                omega_l = 1
            sctx.mats_state_array[0] = omega_l
            sctx.mats_state_array[1] = omega_r

        omega_l = sctx.mats_state_array[0]
        omega_r = sctx.mats_state_array[1]

        if eps_app_eng.shape[0] == 4:
            D_mtx = zeros((4, 4))
            sigma = zeros(4)
        else:
            print("MATS1D5Bond: Unsupported number of strain components")

        if omega_l or eps_app_eng[0] > self.s_cr:
            D_mtx[0, 0] = 0
            sigma[0] = self.tau_fr  # tau_l
        else:
            D_mtx[0, 0] = self.tau_max / self.s_cr
            sigma[0] = D_mtx[0, 0] * eps_app_eng[0]

        if omega_r or eps_app_eng[1] > self.s_cr:
            D_mtx[1, 1] = 0.
            sigma[1] = self.tau_fr  # tau_r
        else:
            D_mtx[1, 1] = self.tau_max / self.s_cr
            sigma[1] = D_mtx[1, 1] * eps_app_eng[1]

        D_mtx[-2, -2] = self.Ef * self.Af
        D_mtx[-1, -1] = self.Em * self.Am

        sigma[-2] = self.Ef * self.Af * eps_app_eng[-2]  # sig_f
        sigma[-1] = self.Em * self.Am * eps_app_eng[-1]  # sig_m

        # You print the stress you just computed and the value of the apparent
        # E

        return sigma, D_mtx

    #-------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # Response trace evaluators
    #-------------------------------------------------------------------------

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app_t1d': self.get_sig_app,
                'eps_app_t1d': self.get_eps_app,
                'shear_s': self.get_shear,
                'slip_s': self.get_slip,
                'msig_pos': self.get_msig_pos,
                'msig_pm': self.get_msig_pm}
