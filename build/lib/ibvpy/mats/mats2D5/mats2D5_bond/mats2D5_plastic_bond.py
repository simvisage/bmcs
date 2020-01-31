'''
Created on Feb 25, 2010

@author: jakub
'''
from math import pi as Pi, cos, sin, exp, sqrt as scalar_sqrt

from ibvpy.api import RTrace, RTDofGraph, RTraceArraySnapshot
from ibvpy.core.tstepper import \
     TStepper as TS
from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats_eval import IMATSEval
from numpy import \
     array, ones, zeros, outer, inner, transpose, dot, frompyfunc, \
     fabs, sqrt, linspace, vdot, identity, tensordot, \
     sin as nsin, meshgrid, float_, ix_, \
     vstack, hstack, sqrt as arr_sqrt, sign
from scipy.linalg import eig, inv
from traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, \
     Instance, Int, Trait, Range, HasTraits, on_trait_change, Event, \
     Dict, Property, cached_property, Delegate
from traitsui.api import \
     Item, View, HSplit, VSplit, VGroup, Group, Spring


# from dacwt import DAC
#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS2D5PlasticBond(MATS2DEval):
    '''
    Elastic Model.
    '''

    # implements(IMATSEval)

    #---------------------------------------------------------------------------
    # Parameters of the numerical algorithm (integration)
    #---------------------------------------------------------------------------

    stress_state = Enum("plane_stress", "plane_strain")

    #---------------------------------------------------------------------------
    # Material parameters 
    #---------------------------------------------------------------------------

    E_m = Float(1.,  # 34e+3,
                 label="E_m",
                 desc="Young's Modulus",
                 auto_set=False)
    nu_m = Float(0.2,
                 label='nu_m',
                 desc="Poison's ratio",
                 auto_set=False)

    E_f = Float(1.,  # 34e+3,
                 label="E_f",
                 desc="Young's Modulus",
                 auto_set=False)
    nu_f = Float(0.2,
                 label='nu_f',
                 desc="Poison's ratio",
                 auto_set=False)

    G = Float(1.,  # 34e+3,
                 label="G",
                 desc="Shear Modulus",
                 auto_set=False)

    sigma_y = Float(.5,  # 34e+3,
                label="s_y",
                desc="Yield stress",
                auto_set=False)

    K_bar = Float(0.,  # 34e+3,
                label="K",
                desc="isotropic hardening",
                auto_set=False)

    H_bar = Float(0.,  # 34e+3,
                   label="H",
                   desc="kinematic hardening",
                   auto_set=False)

    D_el = Property(Array(float), depends_on='E_f, nu_f,E_m,nu_f,G, stress_state')

    @cached_property
    def _get_D_el(self):
        if self.stress_state == "plane_stress":
            return self._get_D_plane_stress()
        else:
            return self._get_D_plane_strain()

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = Event

    #---------------------------------------------------------------------------------------------
    # View specification
    #---------------------------------------------------------------------------------------------

    view_traits = View(VSplit(Group(Item('E_m'),
                                      Item('nu_m'),
                                      Item('E_f'),
                                      Item('nu_f'),
                                      Item('G')
                                      ),
                                Group(Item('stress_state', style='custom'),
                                       Spring(resizable=True),
                                       label='Configuration parameters', show_border=True,
                                       ),
                                ),
                        resizable=True
                        )

    #-----------------------------------------------------------------------------------------------
    # Private initialization methods
    #-----------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-----------------------------------------------------------------------------------------------

    def new_cntl_var(self):
        return zeros(3, float_)

    def new_resp_var(self):
        return zeros(3, float_)

    def get_state_array_size(self):
        return 3

    #-----------------------------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-----------------------------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        sigma = dot(self.D_el[:], eps_app_eng)

        # You print the stress you just computed and the value of the apparent E
        eps_n1 = float(eps_app_eng[6])  # hack for this particular case
        G = self.G

        eps_avg = eps_n1

        if sctx.update_state_on:
            eps_n = eps_avg - float(d_eps[6])
            sctx.mats_state_array[:] = self._get_state_variables(sctx, eps_n)

        # print 'state array ', sctx.mats_state_array
        eps_p_n, q_n, alpha_n = sctx.mats_state_array
        sigma_trial = self.G * (eps_n1 - eps_p_n)
        xi_trial = sigma_trial - q_n
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha_n)
        # f_trial = -xi_trial - ( self.sigma_y + self.K_bar * alpha_n )

        sig_n1 = zeros((1,), dtype='float_')
        D_n1 = zeros((1, 1), dtype='float_')
        if f_trial <= 1e-8:
            sig_n1[0] = sigma_trial
            D_n1[0, 0] = G
        else:
            # print 'plastic'
            d_gamma = f_trial / (self.G + self.K_bar + self.H_bar)
            sig_n1[0] = sigma_trial - d_gamma * self.G * sign(xi_trial)
            D_n1[0, 0] = (self.G * (self.K_bar + self.H_bar)) / \
                            (self.G + self.K_bar + self.H_bar)
            # print 'stress ', sig_n1[0]
        sigma[6] = sig_n1[0]
        self.D_el[6, 6] = D_n1[0, 0]
        return  sigma, self.D_el

    #---------------------------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #---------------------------------------------------------------------------------------------

    def _get_state_variables(self, sctx, eps_n):

        eps_p_n, q_n, alpha_n = sctx.mats_state_array

        # Get the characteristics of the trial step
        #
        sig_trial = self.G * (eps_n - eps_p_n)
        xi_trial = sig_trial - q_n
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha_n)

        if f_trial > 1e-8 :

            #
            # Tha last equilibrated step was inelastic. Here the
            # corresponding state variables must be calculated once
            # again. This might be expensive for 2D and 3D models. Then,
            # some kind of caching should be considered for the state
            # variables determined during iteration. In particular, the
            # computation of d_gamma should be outsourced into a separate
            # method that can in general perform an iterative computation.
            #
            d_gamma = f_trial / (self.G + self.K_bar + self.H_bar)
            eps_p_n += d_gamma * sign(xi_trial)
            q_n += d_gamma * self.H_bar * sign(xi_trial)
            alpha_n += d_gamma

        newarr = array([eps_p_n, q_n, alpha_n], dtype='float_')

        return newarr

    def _get_D_plane_stress(self):
        E_m = self.E_m
        nu_m = self.nu_m
        E_f = self.E_f
        nu_f = self.nu_f
        G = self.G
        D_stress = zeros([8, 8])
        D_stress[0, 0] = E_m / (1.0 - nu_m * nu_m)
        D_stress[0, 1] = E_m / (1.0 - nu_m * nu_m) * nu_m
        D_stress[1, 0] = E_m / (1.0 - nu_m * nu_m) * nu_m
        D_stress[1, 1] = E_m / (1.0 - nu_m * nu_m)
        D_stress[2, 2] = E_m / (1.0 - nu_m * nu_m) * (1.0 / 2.0 - nu_m / 2.0)

        D_stress[3, 3] = E_f / (1.0 - nu_f * nu_f)
        D_stress[3, 4] = E_f / (1.0 - nu_f * nu_f) * nu_f
        D_stress[4, 3] = E_f / (1.0 - nu_f * nu_f) * nu_f
        D_stress[4, 4] = E_f / (1.0 - nu_f * nu_f)
        D_stress[5, 5] = E_f / (1.0 - nu_f * nu_f) * (1.0 / 2.0 - nu_f / 2.0)

        D_stress[6, 6] = G
        D_stress[7, 7] = G
        return D_stress

    def _get_D_plane_strain(self):
        # TODO: adapt to use arbitrary 2d model following the 1d5 bond
        E_m = self.E_m
        nu_m = self.nu_m
        E_f = self.E_f
        nu_f = self.nu_f
        G = self.G
        D_strain = zeros([8, 8])
        D_strain[0, 0] = E_m * (1.0 - nu_m) / (1.0 + nu_m) / (1.0 - 2.0 * nu_m)
        D_strain[0, 1] = E_m / (1.0 + nu_m) / (1.0 - 2.0 * nu_m) * nu_m
        D_strain[1, 0] = E_m / (1.0 + nu_m) / (1.0 - 2.0 * nu_m) * nu_m
        D_strain[1, 1] = E_m * (1.0 - nu_m) / (1.0 + nu_m) / (1.0 - 2.0 * nu_m)
        D_strain[2, 2] = E_m * (1.0 - nu_m) / (1.0 + nu_m) / (2.0 - 2.0 * nu_m)

        D_strain[3, 3] = E_f * (1.0 - nu_f) / (1.0 + nu_f) / (1.0 - 2.0 * nu_f)
        D_strain[3, 4] = E_f / (1.0 + nu_f) / (1.0 - 2.0 * nu_f) * nu_f
        D_strain[4, 3] = E_f / (1.0 + nu_f) / (1.0 - 2.0 * nu_f) * nu_f
        D_strain[4, 4] = E_f * (1.0 - nu_f) / (1.0 + nu_f) / (1.0 - 2.0 * nu_f)
        D_strain[5, 5] = E_f * (1.0 - nu_f) / (1.0 + nu_f) / (2.0 - 2.0 * nu_f)

        D_strain[6, 6] = G
        D_strain[7, 7] = G
        return D_strain

    #---------------------------------------------------------------------------------------------
    # Response trace evaluators
    #---------------------------------------------------------------------------------------------

    def get_sig_norm(self, sctx, eps_app_eng):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return array([ scalar_sqrt(sig_eng[0] ** 2 + sig_eng[1] ** 2) ])

    def get_eps_app_m(self, sctx, eps_app_eng):
        return self.map_eps_eng_to_mtx((eps_app_eng[:3]))

    def get_eps_app_f(self, sctx, eps_app_eng):
        return self.map_eps_eng_to_mtx((eps_app_eng[3:6]))

    def get_sig_app_m(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return self.map_sig_eng_to_mtx((sig_eng[:3]))

    def get_sig_app_f(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return self.map_sig_eng_to_mtx((sig_eng[3:6]))

    def get_sig_b(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return array([[sig_eng[6], 0.], [0., sig_eng[7]]])

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {
                'eps_app_f'  : self.get_eps_app_f,
                'eps_app_m'  : self.get_eps_app_m,
                'sig_app_f'  : self.get_sig_app_f,
                'sig_app_m'  : self.get_sig_app_m,
                'sig_norm'   : self.get_sig_norm,
                'sig_b'      : self.get_sig_b, }

