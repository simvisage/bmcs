
from math import pi as Pi, cos, sin, exp

from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats3D.vmats3D_eval import MATS3D
from ibvpy.mats.mats_eval import IMATSEval
from numpy import \
    array, zeros, dot, float_
from traits.api import \
    Enum, Float, HasTraits, Enum, \
    Instance, Trait, Range, HasTraits, on_trait_change, Event, \
    implements, Dict, Property, Array, cached_property
from traitsui.api import \
    Item, View, VSplit, Group, Spring
from util.traits.either_type import \
    EitherType

from strain_norm3d import Energy, Euclidean, Mises, Rankine, Mazars, \
    IStrainNorm3D


#from scipy.linalg import eig, inv
#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS3DScalarDamage(MATS3DEval):
    '''
    Scalar Damage Model.
    '''

    implements(IMATSEval)

    #-------------------------------------------------------------------------
    # Parameters of the numerical algorithm (integration)
    #-------------------------------------------------------------------------

    stiffness = Enum("algoritmic", "secant")

    #-------------------------------------------------------------------------
    # Material parameters
    #-------------------------------------------------------------------------

    E = Float(1.,  # 34e+3,
              label="E",
              desc="Young's Modulus",
              auto_set=False)
    nu = Float(0.2,
               label='nu',
               desc="Poison's ratio",
               auto_set=False)

    epsilon_0 = Float(59e-6,
                      label="eps_0",
                      desc="Breaking Strain",
                      auto_set=False)

    epsilon_f = Float(191e-6,
                      label="eps_f",
                      desc="Shape Factor",
                      auto_set=False)

    stiffness = Enum("secant", "algorithmic")

    strain_norm = EitherType(klasses=[Energy,
                                      Euclidean,
                                      Mises,
                                      Rankine,
                                      Mazars])

    D_el = Property(Array(float), depends_on='E, nu')

    @cached_property
    def _get_D_el(self):
        return self._get_D_el()

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = Event

    #--------------------------------------------------------------------------
    # View specification
    #--------------------------------------------------------------------------

    view_traits = View(VSplit(Group(Item('E'),
                                    Item('nu'),
                                    Item('strain_norm')),
                              Group(Item('stiffness', style='custom'),
                                    Spring(resizable=True),
                                    label='Configuration parameters',
                                    show_border=True,
                                    ),
                              ),
                       resizable=True
                       )

    #--------------------------------------------------------------------------
    # Private initialization methods
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #--------------------------------------------------------------------------

    def get_state_array_size(self):
        '''
        Return number of number to be stored in state array
        @param sctx:spatial context
        '''
        return 2

    def setup(self, sctx):
        '''
        Intialize state variables.
        @param sctx:spatial context
        '''
        sctx.mats_state_array[:] = zeros(2, float_)
        #sctx.update_state_on = False

    def new_cntl_var(self):
        '''
        Return contoll variable array
        '''
        return zeros(6, float_)

    def new_resp_var(self):
        '''
        Return control response array
        '''
        return zeros(6, float_)

    #--------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #--------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1, eps_avg=None):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        if eps_avg != None:
            pass
        else:
            eps_avg = eps_app_eng

        if sctx.update_state_on:
            # print "in us"
            eps_n = eps_avg - d_eps

            e_max, omega = self._get_state_variables(sctx, eps_n)

            sctx.mats_state_array[0] = e_max
            sctx.mats_state_array[1] = omega

        e_max, omega = self._get_state_variables(sctx, eps_app_eng)

        if self.stiffness == "algorithmic" and e_max > self.epsilon_0:
            D_e_dam = self._get_alg_stiffness(eps_app_eng,
                                              e_max,
                                              omega)
        else:
            D_e_dam = (1 - omega) * self.D_el

        sigma = dot(((1 - omega) * self.D_el), eps_app_eng)

        # You print the stress you just computed and the value of the apparent
        # E

        return sigma, D_e_dam

    #--------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #--------------------------------------------------------------------------
    def _get_state_variables(self, sctx, eps_app_eng):
        e_max = sctx.mats_state_array[0]
        omega = sctx.mats_state_array[1]

        f_trial = self.strain_norm.get_f_trial(eps_app_eng,
                                               self.D_el,
                                               self.E,
                                               self.nu,
                                               e_max)
        if f_trial > 0:
            e_max += f_trial
            omega = self._get_omega(e_max)

        return e_max, omega

    def _get_D_el(self):
        '''
        Return elastic stiffness matrix
        '''
        D_el = zeros((6, 6))
        t2 = 1. / (1. + self.nu)
        t3 = self.E * t2
        t9 = self.E * self.nu * t2 / (1. - 2. * self.nu)
        t10 = t3 + t9
        t11 = t3 / 2.
        D_el[0][0] = t10
        D_el[0][1] = t9
        D_el[0][2] = t9
        D_el[0][3] = 0.
        D_el[0][4] = 0.
        D_el[0][5] = 0.
        D_el[1][0] = t9
        D_el[1][1] = t10
        D_el[1][2] = t9
        D_el[1][3] = 0.
        D_el[1][4] = 0.
        D_el[1][5] = 0.
        D_el[2][0] = t9
        D_el[2][1] = t9
        D_el[2][2] = t10
        D_el[2][3] = 0.
        D_el[2][4] = 0.
        D_el[2][5] = 0.
        D_el[3][0] = 0.
        D_el[3][1] = 0.
        D_el[3][2] = 0.
        D_el[3][3] = t11
        D_el[3][4] = 0.
        D_el[3][5] = 0.
        D_el[4][0] = 0.
        D_el[4][1] = 0.
        D_el[4][2] = 0.
        D_el[4][3] = 0.
        D_el[4][4] = t11
        D_el[4][5] = 0.
        D_el[5][0] = 0.
        D_el[5][1] = 0.
        D_el[5][2] = 0.
        D_el[5][3] = 0.
        D_el[5][4] = 0.
        D_el[5][5] = t11
        return D_el

    def _get_omega(self, kappa):
        '''
        Return new value of damage parameter
        @param kappa:
        '''
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        if kappa >= epsilon_0:
            return 1. - epsilon_0 / kappa * exp(-1 * (kappa - epsilon_0) / epsilon_f)
        else:
            return 0.

    def _get_alg_stiffness(self, eps_app_eng, e_max, omega):
        '''
        Return algorithmic stiffness matrix
        @param eps_app_eng:strain
        @param e_max:kappa
        @param omega:damage parameter
        '''
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        dodk = epsilon_0 / (e_max * e_max) * exp(-(e_max - epsilon_0) / epsilon_f) + \
            epsilon_0 / e_max / epsilon_f * \
            exp(-(e_max - epsilon_0) / epsilon_f)
        dede = self.strain_norm.get_dede(
            eps_app_eng, self.D_el, self.E, self.nu)
        D_alg = (1 - omega) * self.D_el - \
            dot(dot(self.D_el, eps_app_eng), dede) * dodk
        return D_alg

    #--------------------------------------------------------------------------
    # Response trace evaluators
    #--------------------------------------------------------------------------

    def get_omega(self, sctx, eps_app_eng):
        '''
        Return damage parameter for RT
        @param sctx:spatial context
        @param eps_app_eng:actual strain
        '''
        return array([sctx.mats_state_array[1]])

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'omega': self.get_omega}
