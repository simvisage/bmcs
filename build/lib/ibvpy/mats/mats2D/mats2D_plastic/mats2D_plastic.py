
from math import pi as Pi, cos, sin, exp, sqrt as scalar_sqrt

from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from ibvpy.mats.mats_eval import IMATSEval
from numpy import \
    array, zeros, dot, \
    sqrt, vdot, \
    float_, diag
from scipy.linalg import eig, inv, norm
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, \
    Instance, Int, Trait, Range, HasTraits, on_trait_change, Event, \
    Dict, Property, cached_property, Delegate
from traitsui.api import \
    Item, View, HSplit, VSplit, VGroup, Group, Spring
from util.traits.either_type import EitherType

import numpy as np

from .yield_face2D import IYieldFace2D, J2, DruckerPrager, Gurson, CamClay


#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS2DPlastic(MATS2DEval):
    '''
    Elastic Model.
    '''

    # implements(IMATSEval)

    #-------------------------------------------------------------------------
    # Parameters of the numerical algorithm (integration)
    #-------------------------------------------------------------------------

    stress_state = Enum("plane strain", "plane stress",)
    algorithm = Enum("closest point", "cutting plane")

    #-------------------------------------------------------------------------
    # Material parameters
    #-------------------------------------------------------------------------
    yf = EitherType(klasses=[J2, DruckerPrager, Gurson, CamClay],
                    label="Yield Face",
                    desc="Yield Face Definition"
                    )
#    yf = Instance( IYieldFace2D,
#                 label = "Yield Face",
#                 desc = "Yield Face Definition")
    E = Float(210.0e+3,
              label="E",
              desc="Young's Modulus")
    nu = Float(0.2,
               label='nu',
               desc="Poison's ratio")
    K_bar = Float(0.,
                  label='K',
                  desc="isotropic softening parameter")
    H_bar = Float(0.,
                  label='H',
                  desc="kinematic softening parameter")
    tolerance = Float(1.0e-4,
                      label='TOL',
                      desc="tolerance of return mapping")

    max_iter = Int(20,
                   label='Iterations',
                   desc="maximal number of iterations")

    D_el = Property(Array(float), depends_on='E, nu')

    @cached_property
    def _get_D_el(self):
        if self.stress_state == "plane_stress":
            return self._get_D_plane_stress()
        else:
            return self._get_D_plane_strain()

    H_mtx = Property(Array(float), depends_on='K_bar, H_bar')

    @cached_property
    def _get_H_mtx(self):
        H_mtx = diag([self.K_bar, self.H_bar, self.H_bar, self.H_bar])
        return H_mtx

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #

    #-------------------------------------------------------------------------
    # View specification
    #-------------------------------------------------------------------------

    view_traits = View(VSplit(Group(Item('yf'),
                                    Item('E'),
                                    Item('nu'),
                                    Item('K_bar'),
                                    Item('H_bar'),
                                    Item('tolerance'),
                                    Item('max_iter')),
                              Group(Item('stress_state', style='custom'),
                                    Item('algorithm', style='custom'),
                                    Spring(resizable=True),
                                    label='Configuration parameters', show_border=True,
                                    ),
                              ),
                       resizable=True
                       )

    #-------------------------------------------------------------------------
    # Private initialization methods
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------
    def get_state_array_size(self):
        '''
        Return number of number to be stored in state array
        @param sctx:spatial context
        '''
        return 7

    def setup(self, sctx):
        '''
        Intialize state variables.
        '''
        sctx.mats_state_array = zeros(7, float_)

    def new_cntl_var(self):
        return zeros(3, float_)

    def new_resp_var(self):
        return zeros(3, float_)

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        delta_gamma = 0.
        if sctx.update_state_on:
            # print "in us"
            eps_n = eps_app_eng - d_eps
            sigma, f_trial, epsilon_p, q_1, q_2 = self._get_state_variables(
                sctx, eps_n)

            sctx.mats_state_array[:3] = epsilon_p
            sctx.mats_state_array[3] = q_1
            sctx.mats_state_array[4:] = q_2

        diff1s = zeros([3])
        sigma, f_trial, epsilon_p, q_1, q_2 = self._get_state_variables(
            sctx, eps_app_eng)
        # Note: the state variables are not needed here, just gamma
        diff2ss = self.yf.get_diff2ss(eps_app_eng, self.E, self.nu, sctx)
        Xi_mtx = inv(inv(self.D_el) + delta_gamma * diff2ss * f_trial)
        N_mtx_denom = sqrt(dot(dot(diff1s, Xi_mtx), diff1s))
        if N_mtx_denom == 0.:
            N_mtx = zeros(3)
        else:
            N_mtx = dot(Xi_mtx, self.diff1s) / N_mtx_denom
        D_mtx = Xi_mtx - vdot(N_mtx, N_mtx)

        # print "sigma ",sigma
        # print "D_mtx ",D_mtx
        return sigma, D_mtx

    #-------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #-------------------------------------------------------------------------
    def _get_state_variables(self, sctx, eps_app_eng):
        epsilon_p = eps_p_old = sctx.mats_state_array[:3]
        q_1 = sctx.mats_state_array[3]
        q_2 = sctx.mats_state_array[4:]

        sigma = dot(self.D_el, (eps_app_eng - epsilon_p))
        xi_trial = sigma - q_2
        f_trial = self.yf.get_f_trial(xi_trial, q_1)
        # print "f_trial", f_trial
        R_k = zeros(3)
        int_count = 1
        while f_trial > self.tolerance or norm(R_k) > self.tolerance:
            if int_count > self.max_iter:
                print("Maximal number of iteration reached")
                break
            diff1s = self.yf.get_diff1s(
                eps_app_eng, self.E, self.nu, sctx)
            diff1q = self.yf.get_diff1q(eps_app_eng, self.E, self.nu, sctx)

            if self.stress_state == "plane stress":
                raise NotImplementedError
            else:
                if self.algorithm == "cutting plane":
                    delta_gamma_2 = (f_trial / 
                                     (dot(dot(diff1s, self.D_el), diff1s) + 
                                      dot(dot(diff1q, self.H_mtx), diff1q)))

                    delta_gamma += delta_gamma_2
                    epsilon_p += delta_gamma_2 * diff1s

                    q_1 += delta_gamma_2 * self.K_bar * diff1q[0]
                    q_2 += delta_gamma_2 * self.H_bar * diff1q[1:]

                elif self.algorithm == "closest point":
                    diff2ss = self.yf.get_diff2ss(
                        eps_app_eng, self.E, self.nu, sctx)
                    diff2sq = self.yf.get_diff2sq(
                        eps_app_eng, self.E, self.nu, sctx)
                    A_mtx_inv = inv(self.D_el) + delta_gamma * diff2ss
                    # self.delta_gamma *  diff2sq))
                    A_mtx = inv(A_mtx_inv)
                    delta_gamma_2 = ((f_trial - dot(dot(diff1s, A_mtx), R_k)) / 
                                     (dot(dot(diff1s, A_mtx), diff1s)))

                    delta_eps_p = dot(dot(inv(self.D_el), A_mtx),
                                      R_k + delta_gamma_2 * diff1s)

                    epsilon_p += delta_eps_p
                    delta_gamma += delta_gamma_2

                    R_k = -epsilon_p + eps_p_old + delta_gamma * diff1s

                else:
                    raise NotImplementedError

            sigma = dot(self.D_el, (eps_app_eng - epsilon_p))
            xi_trial = sigma - q_2

            f_trial = self.yf.get_f_trial(xi_trial, q_1)
            # print "f_trial_after", self.f_trial
            int_count += 1
        return sigma, f_trial, epsilon_p, q_1, q_2

    def _get_D_plane_stress(self):
        E = self.E
        nu = self.nu
        D_stress = zeros([3, 3])
        D_stress[0][0] = E / (1.0 - nu * nu)
        D_stress[0][1] = E / (1.0 - nu * nu) * nu
        D_stress[1][0] = E / (1.0 - nu * nu) * nu
        D_stress[1][1] = E / (1.0 - nu * nu)
        D_stress[2][2] = E / (1.0 - nu * nu) * (1.0 / 2.0 - nu / 2.0)
        return D_stress

    def _get_D_plane_strain(self):
        E = self.E
        nu = self.nu
        D_strain = zeros([3, 3])
        D_strain[0][0] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[0][1] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1][0] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1][1] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[2][2] = E * (1.0 - nu) / (1.0 + nu) / (2.0 - 2.0 * nu)
        return D_strain

    #-------------------------------------------------------------------------
    # Response trace evaluators
    #-------------------------------------------------------------------------

    def get_sig_norm(self, sctx, eps_app_eng):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return array([scalar_sqrt(sig_eng[0] ** 2 + sig_eng[1] ** 2)])

    def get_eps_p(self, sctx, eps_app_eng):
        # print "eps tracer ", sctx.mats_state_array[:3]
        return sctx.mats_state_array[:3]

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'eps_app': self.get_eps_app,
                'sig_norm': self.get_sig_norm,
                'eps_p': self.get_eps_p}


if __name__ == '__main__':
    #-------------------------------------------------------------------------
    # Example using the mats2d_explore
    #-------------------------------------------------------------------------
    from ibvpy.api import RTDofGraph
    from ibvpy.mats.mats2D.mats2D_explore import MATS2DExplore
    from .yield_face2D import J2
    mats2D_explore = \
        MATS2DExplore(mats2D_eval=MATS2DPlastic(yf=J2()),
                      rtrace_list=[RTDofGraph(name='strain 0 - stress 0',
                                               var_x='eps_app', idx_x=0,
                                               var_y='sig_app', idx_y=0,
                                               record_on='update'),
                                   RTDofGraph(name='strain 0 - strain 1',
                                               var_x='eps_app', idx_x=0,
                                               var_y='eps_app', idx_y=1,
                                               record_on='update'),
                                   RTDofGraph(name='stress 0 - stress 1',
                                               var_x='sig_app', idx_x=0,
                                               var_y='sig_app', idx_y=1,
                                               record_on='update'),
                                   RTDofGraph(name='time - sig_norm',
                                               var_x='time', idx_x=0,
                                               var_y='sig_norm', idx_y=0,
                                               record_on='update')

                                   ])

    mats2D_explore.tloop.eval()
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp(ibv_resource=mats2D_explore)
    ibvpy_app.main()
#    from ibvpy.core.sdomain import SDomain
#    mm = MATS2DPlastic(E = 1.,
#                       nu = 0.,
#                       yf = J2(sigma_y = 1.))
#
#    eps_app_eng = array([1.5,0.,0.])
#    d_eps = array([1.,0.,0.])
#    sctx = SDomain()
#    sctx.update_state_on = False
#    sctx.mats_state_array = zeros(7)
#    tn = tn1 = 0.
#
#    sig, D = mm.get_corr_pred(sctx, eps_app_eng, d_eps, tn, tn1)
#
#    print sig
