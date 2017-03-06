
from numpy import \
    array, ones, zeros, outer, inner, transpose, dot, frompyfunc, \
    fabs, sqrt, linspace, vdot, identity, tensordot, \
    sin as nsin, meshgrid, float_, ix_, \
    vstack, hstack, sqrt as arr_sqrt
from scipy.linalg import eig, inv
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, \
    Instance, Int, Trait, Range, HasTraits, on_trait_change, Event, \
    implements, Dict, Property, cached_property, Delegate
from traitsui.api import \
    Item, View, HSplit, VSplit, VGroup, Group, Spring

from ibvpy.core.tstepper import \
    TStepper as TS
from ibvpy.mats.mats1D.mats1D_eval import MATS1DEval
from ibvpy.mats.mats_eval import IMATSEval
from mathkit.mfn import MFnLineArray


#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS1DElastic(MATS1DEval):
    '''
    Elastic Model.
    '''

    implements(IMATSEval)

    E = Float(1.,  # 34e+3,
              label="E",
              desc="Young's Modulus",
              auto_set=False, enter_set=True)

    #-------------------------------------------------------------------------
    # Piece wise linear stress strain curve
    #-------------------------------------------------------------------------
    _stress_strain_curve = Instance(MFnLineArray)

    def __stress_strain_curve_default(self):
        return MFnLineArray(ydata=[0., self.E],
                            xdata=[0., 1.])

    @on_trait_change('E')
    def reset_stress_strain_curve(self):
        self._stress_strain_curve = MFnLineArray(ydata=[0., self.E],
                                                 xdata=[0., 1.])

    stress_strain_curve = Property

    def _get_stress_strain_curve(self):
        return self._stress_strain_curve

    def _set_stress_strain_curve(self, curve):
        self._stress_strain_curve = curve

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = Event

    #-------------------------------------------------------------------------
    # View specification
    #-------------------------------------------------------------------------

    view_traits = View(Item('E'),
                       )

    #-------------------------------------------------------------------------
    # Private initialization methods
    #-------------------------------------------------------------------------
    def __init__(self, **kwtraits):
        '''
        Subsidiary arrays required for the integration.
        they are set up only once prior to the computation
        '''
        super(MATS1DElastic, self).__init__(**kwtraits)

    def new_cntl_var(self):
        return zeros(1, float_)

    def new_resp_var(self):
        return zeros(1, float_)

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1, *args, **kw):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        eps_n1 = float(eps_app_eng)
#        E   = self.E
#        D_el = array([[E]])
#        sigma = dot( D_el, eps_app_eng )
        D_el = array([[self.stress_strain_curve.get_diff(eps_n1)]])
        sigma = array([self.stress_strain_curve.get_value(eps_n1)])
        # You print the stress you just computed and the value of the apparent
        # E
        return sigma, D_el

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
        return {'sig_app': self.get_sig_app,
                'eps_app': self.get_eps_app}

    #-------------------------------------------------------------------------
    # List of response tracers to be constructed within the mats_explorer
    #-------------------------------------------------------------------------
    def _get_explorer_rtrace_list(self):
        '''Return the list of relevant tracers to be used in mats_explorer.
        '''
        return [
            RTraceGraph(name='strain - stress',
                        var_x='eps_app', idx_x=0,
                        var_y='sig_app', idx_y=0,
                        record_on='update')
        ]

if __name__ == '__main__':
    #-------------------------------------------------------------------------
    # Example
    #-------------------------------------------------------------------------

    from ibvpy.core.tloop import TLoop, TLine
    from ibvpy.api import BCDof
    from ibvpy.core.ibvp_solve import IBVPSolve as IS
    from ibvpy.api import RTraceGraph
    # tseval for a material model
    #
    tseval = MATS1DElastic()
    ts = TS(tse=tseval,
            bcond_list=[BCDof(var='u', dof=0, value=1.)
                        ],
            rtrace_list=[RTraceGraph(name='strain 0 - stress 0',
                                     var_x='eps_app', idx_x=0,
                                     var_y='sig_app', idx_y=0,
                                     record_on='update')
                         ]
            )

    # Put the time-stepper into the time-loop
    #

    tmax = 4.0
    # tmax = 0.0006
    n_steps = 100

    tl = TLoop(tstepper=ts,
               DT=tmax / n_steps, KMAX=100, RESETMAX=0,
               tline=TLine(min=0.0, max=tmax))

    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    tl.eval()

    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(tloop=tl)
    app.main()
