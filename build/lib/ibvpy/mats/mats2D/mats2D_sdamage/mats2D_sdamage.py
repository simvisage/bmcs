
from math import pi as Pi, cos, sin, exp

from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from ibvpy.mats.mats_eval import IMATSEval
from numpy import \
    array, zeros, dot, float_
from traits.api import \
    Enum, Float, HasTraits, Enum, \
    Instance, Trait, Range, HasTraits, on_trait_change, Event, \
    implements, Dict, Property, cached_property, Array
from traitsui.api import \
    Item, View, VSplit, Group, Spring
from util.traits.either_type import EitherType

from strain_norm2d import Energy, Euclidean, Mises, Rankine, Mazars, \
    IStrainNorm2D


# from scipy.linalg import eig, inv
#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS2DScalarDamage(MATS2DEval):
    '''
    Scalar Damage Model.
    '''

    implements(IMATSEval)

    #-------------------------------------------------------------------------
    # Parameters of the numerical algorithm (integration)
    #-------------------------------------------------------------------------

    stress_state = Enum("plane_stress", "plane_strain")
    stiffness = Enum("secant", "algoritmic")

    #-------------------------------------------------------------------------
    # Material parameters
    #-------------------------------------------------------------------------

    E = Float(34e+3,
              label="E",
              desc="Young's Modulus",
              auto_set=False)
    nu = Float(0.2,
               label='nu',
               desc="Poison's ratio",
               auto_set=False)
    epsilon_0 = Float(59e-6,
                      label="eps_0",
                      desc="Strain at the onset of damage",
                      auto_set=False)

    epsilon_f = Float(191e-4,
                      label="eps_f",
                      desc="Slope of the damage function",
                      auto_set=False)

    strain_norm = EitherType(klasses=[Mazars,
                                      Euclidean,
                                      Energy,
                                      Mises,
                                      Rankine])

    D_el = Property(Array(float), depends_on='E, nu, stress_state')

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

    #--------------------------------------------------------------------------
    # View specification
    #--------------------------------------------------------------------------

    view_traits = View(VSplit(Group(Item('E'),
                                    Item('nu'),
                                    Item('epsilon_0'),
                                    Item('epsilon_f'),
                                    Item('strain_norm')),
                              Group(Item('stress_state', style='custom'),
                                    Item('stiffness', style='custom'),
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
        state_arr_size = self.get_state_array_size()
        sctx.mats_state_array = zeros(state_arr_size, 'float_')
        # sctx.update_state_on = False

    def new_cntl_var(self):
        '''
        Return contoll variable array
        '''
        return zeros(3, float_)

    def new_resp_var(self):
        '''
        Return contoll response array
        '''
        return zeros(3, float_)

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

            eps_n = eps_avg - d_eps

            e_max, omega = self._get_state_variables(sctx, eps_n)

            sctx.mats_state_array[0] = e_max
            sctx.mats_state_array[1] = omega

        e_max, omega = self._get_state_variables(sctx, eps_app_eng)

        if self.stiffness == "algorithmic" and \
                e_max > self.epsilon_0 and \
                e_max > sctx.mats_state_array[0]:
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

    def _get_omega(self, kappa):
        '''
        Return new value of damage parameter
        @param kappa:
        '''
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        if kappa >= epsilon_0:
            # return 1.-epsilon_0/kappa*exp(-1*(kappa-epsilon_0)/epsilon_f)
            return 1. - epsilon_0 / kappa * exp(-1 * (kappa - epsilon_0) /
                                                (epsilon_f - epsilon_0))
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

    def _get_D_plane_stress(self):
        '''
        Elastic Matrix - Plane Stress
        '''
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
        '''
        Elastic Matrix - Plane Strain
        '''
        E = self.E
        nu = self.nu
        D_strain = zeros([3, 3])
        D_strain[0][0] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[0][1] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1][0] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1][1] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[2][2] = E * (1.0 - nu) / (1.0 + nu) / (2.0 - 2.0 * nu)
        return D_strain

    #--------------------------------------------------------------------------
    # Response trace evaluators
    #--------------------------------------------------------------------------

    def get_omega(self, sctx, eps_app_eng, *args, **kw):
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
                'eps_app': self.get_eps_app,
                'omega': self.get_omega}


if __name__ == '__main__':

    #-------------------------------------------------------------------------
    # Example
    #-------------------------------------------------------------------------

    from ibvpy.api import RTDofGraph
    from ibvpy.mats.mats2D.mats2D_explore import MATS2DExplore

    mats2D_explore = \
        MATS2DExplore(mats2D_eval=MATS2DScalarDamage(strain_norm_type='Rankine'),
                      # stiffness = 'algorithmic' ),
                      rtrace_list=[RTDofGraph(name='strain - stress',
                                              var_x='eps_app', idx_x=0,
                                              var_y='sig_app', idx_y=0,
                                              record_on='update'),
                                   RTDofGraph(name='strain - strain',
                                              var_x='eps_app', idx_x=0,
                                              var_y='eps_app', idx_y=1,
                                              record_on='update'),
                                   RTDofGraph(name='stress - stress',
                                              var_x='sig_app', idx_x=0,
                                              var_y='sig_app', idx_y=1,
                                              record_on='update'),
                                   #                             RTDofGraph(name = 'time - sig_norm',
                                   #                                      var_x = 'time', idx_x = 0,
                                   #                                      var_y = 'sig_norm', idx_y = 0,
                                   # record_on = 'update' ),
                                   ]
                      )

    mats2D_explore.tloop.eval()
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp(ibv_resource=mats2D_explore)
    ibvpy_app.main()
