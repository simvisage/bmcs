from math import pi as Pi, cos, sin, exp, sqrt as scalar_sqrt
from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from ibvpy.mats.mats_eval import IMATSEval
from numpy import array, zeros, dot, float_, copy
from scipy.linalg import eig, inv
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, \
    Trait, Range, HasTraits, Event, \
    Dict, Property, cached_property, Constant, Tuple
from traitsui.api import \
    Item, View, VSplit, Group, Spring
from view.ui import BMCSLeafNode


#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS2DElastic(MATS2DEval, BMCSLeafNode):

    '''
    Elastic Model.
    '''

    # implements(IMATSEval)

    #-------------------------------------------------------------------------
    # Parameters of the numerical algorithm (integration)
    #-------------------------------------------------------------------------

    stress_state = Enum("plane_stress", "plane_strain", "rotational_symetry")

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

    n_s = Constant(4)

    state_arr_shape = Tuple((4,))

    D_el = Property(Array(float), depends_on='E, nu, stress_state')

    @cached_property
    def _get_D_el(self):
        if self.stress_state == "plane_stress":
            return self._get_D_plane_stress()
        elif self.stress_state == "plane_strain":
            return self._get_D_plane_strain()
        elif self.stress_state == "rotational_symetry":
            return self._get_D_rotational_symetry()

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = Event

    #-------------------------------------------------------------------------
    # View specification
    #-------------------------------------------------------------------------

    view_traits = View(VSplit(Group(Item('E'),
                                    Item('nu'),),
                              Group(Item('stress_state', style='custom'),
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
        sigma = dot(self.D_el[:], eps_app_eng)

        # You print the stress you just computed and the value of the apparent
        # E
        return sigma, copy(self.D_el)

    #-------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #-------------------------------------------------------------------------

    def _get_D_plane_stress(self):
        E = self.E
        nu = self.nu
        D_stress = zeros([3, 3])
        D_stress[0, 0] = E / (1.0 - nu * nu)
        D_stress[0, 1] = E / (1.0 - nu * nu) * nu
        D_stress[1, 0] = E / (1.0 - nu * nu) * nu
        D_stress[1, 1] = E / (1.0 - nu * nu)
        D_stress[2, 2] = E / (1.0 - nu * nu) * (1.0 / 2.0 - nu / 2.0)
        return D_stress

    def _get_D_plane_strain(self):
        E = self.E
        nu = self.nu
        D_strain = zeros([3, 3])
        D_strain[0, 0] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[0, 1] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1, 0] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1, 1] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[2, 2] = E * (1.0 - nu) / (1.0 + nu) / (2.0 - 2.0 * nu)
        return D_strain

    def _get_D_rotational_symetry(self):
        E = self.E
        nu = self.nu
        D_strain = zeros([6, 6])
        C = E / (1. - 2. * nu) / (1. + nu)
        D_strain[0, 0] = C * (1. - nu)
        D_strain[0, 1] = C * nu
        D_strain[1, 0] = C * nu
        D_strain[0, 2] = C * nu
        D_strain[2, 0] = C * nu
        D_strain[1, 1] = C * (1. - nu)
        D_strain[1, 2] = C * nu
        D_strain[2, 1] = C * nu
        D_strain[2, 2] = C * (1. - nu)
        D_strain[5, 5] = C * (1.0 - 2 * nu) / 2.
        return D_strain

    #-------------------------------------------------------------------------
    # Response trace evaluators
    #-------------------------------------------------------------------------

    def get_sig_norm(self, sctx, eps_app_eng):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return array([scalar_sqrt(sig_eng[0] ** 2 + sig_eng[1] ** 2)])

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict, transient=True)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'eps_app': self.get_eps_app,
                'sig_norm': self.get_sig_norm,
                'max_principle_sig': self.get_max_principle_sig,
                'strain_energy': self.get_strain_energy}

    def _get_explorer_config(self):
        '''Get the specific configuration of this material model in the explorer
        '''
        c = super(MATS2DElastic, self)._get_explorer_config()

        from ibvpy.api import TLine
        c['tline'] = TLine(step=1.0, max=1.0)
        return c


if __name__ == '__main__':
    #-------------------------------------------------------------------------
    # Example using the mats2d_explore
    #-------------------------------------------------------------------------
    from ibvpy.api import RTDofGraph
    from ibvpy.mats.mats2D.mats2D_explore import MATS2DExplore
    mats2D_explore = \
        MATS2DExplore(mats2D_eval=MATS2DElastic(),
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
    # mme.configure_traits( view = 'traits_view_begehung' )
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp(ibv_resource=mats2D_explore)
    ibvpy_app.main()
