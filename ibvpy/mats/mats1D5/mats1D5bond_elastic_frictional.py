
from math import pi as Pi, cos, sin, exp, sqrt as scalar_sqrt

from ibvpy.api import RTrace, RTDofGraph, RTraceArraySnapshot
from ibvpy.core.tstepper import \
     TStepper as TS
from ibvpy.mats.mats_eval import IMATSEval, MATSEval
from mathkit.mfn import MFnLineArray
from mathkit.mfn.mfn_line.mfn_matplotlib_editor import MFnMatplotlibEditor
from mathkit.mfn.mfn_line.mfn_plot_adapter import MFnPlotAdapter
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

# from dacwt import DAC
mpl_matplotlib_editor = MFnMatplotlibEditor(
                    adapter=MFnPlotAdapter(label_x='slip',
                                              label_y='shear stress'))

#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------


class MATS1D5Bond(MATSEval):
    '''
    Scalar Damage Model.
    '''
    # implements(IMATSEval)

    Ef = Float(1.,  # 34e+3,
                 label="E",
                 desc="Young's Modulus Fiber",
                 auto_set=False, enter_set=True)
    Af = Float(1.,  # 34e+3,
                 label="A",
                 desc="Cross Section Fiber",
                 auto_set=False, enter_set=True)
    Em = Float(1.,  # 34e+3,
                 label="E",
                 desc="Young's Modulus Matrix",
                 auto_set=False, enter_set=True)
    Am = Float(1.,  # 34e+3,
                 label="A",
                 desc="Cross Section Matrix",
                 auto_set=False, enter_set=True)

    s_cr = Float(1.,  # 34e+3,
                 label="s_cr",
                 desc="Critical slip",
                 auto_set=False, enter_set=True)

    perimeter = Property(depends_on='A')

    def _get_perimeter(self):
        return sqrt(4 * self.Af * Pi)

    slip_middle = Bool
    #---------------------------------------------------------------------------------
    # Bond stress (unit area)
    #---------------------------------------------------------------------------------
    tau_max = Float(1.,  # 34e+3,
                 label="tau_max",
                 desc="maximum shear stress",
                 auto_set=False, enter_set=True)
    tau_fr = Float(1.,  # 34e+3,
                 label="tau_fr",
                 desc="Frictional shear stress",
                 auto_set=False, enter_set=True)

    #---------------------------------------------------------------------------------
    # Bond flow (per unit length) - under the assumption of circular cross section
    #---------------------------------------------------------------------------------
    T_max = Property(depends_on='tau_max, Af')

    @cached_property
    def _get_T_max(self):
        return self.tau_max * self.perimeter

    T_fr = Property(depends_on='tau_fr, Af')

    @cached_property
    def _get_T_fr(self):
        return self.tau_fr * self.perimeter

    #---------------------------------------------------------------------------------
    # Bond flow (per unit length) - under the assumption of circular cross section
    #---------------------------------------------------------------------------------
    bond_fn = Property(depends_on='s_cr, tau_max, tau_fr, Af')

    @cached_property
    def _get_bond_fn(self):
        return MFnLineArray(ydata=[ 0., self.T_max, self.T_fr, self.T_fr ],
                             xdata=[ 0., self.s_cr, self.s_cr, self.s_cr * 10])

    traits_view = View(Item('Ef'),
                        Item('Af'),
                        Item('Em'),
                        Item('Am'),
                        Item('tau_fr'),
                        Item('s_cr'),
                        Item('tau_max'),
                        Item('bond_fn', editor=mpl_matplotlib_editor),
                        resizable=True,
                        scrollable=True,
                        height=0.5,
                        width=0.5,
                        buttons=['OK', 'Cancel']
                        )

    def _bond_fn_default(self):
        return MFnLineArray(xdata=[0., 1.], ydata=[0., 1.])

   # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = Event

    #---------------------------------------------------------------------------------------------
    # View specification
    #---------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------
    # Private initialization methods
    #-----------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-----------------------------------------------------------------------------------------------

    def setup(self, sctx):
        '''
        Intialize state variables.
        '''
        sctx.mats_state_array = zeros(sctx.slip_comp, float_)

    def get_state_array_size(self):
        '''
        Give back the nuber of floats to be saved
        @param sctx:spatial context
        '''
        return 2

    def new_cntl_var(self):
        return zeros(4, float_)  # TODO: adapt for 4-5..

    def new_resp_var(self):
        return zeros(4, float_)  # TODO: adapt for 4-5..

    #-----------------------------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-----------------------------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain

        @todo: [Jakub]
        - Describe the state variable vector - what is at what position
        - Describe the components of the strain tensor
        - In all algebraic expressions leave spaces between operators and operands
          a = b and not a=b
        - Comment every relational condition
          if dbond[1] > 0:
        '''
        if eps_app_eng.shape[0] == 5: self.slip_middle = True

        if sctx.update_state_on:
#            print 'in us'
#            print 'eps_app_eng ',eps_app_eng
#            print 'd_eps ',d_eps
            eps_n = eps_app_eng - d_eps
            sctx.mats_state_array[:] = self._get_state_variables(sctx, eps_n)

        dbond = self._get_state_variables(sctx, eps_app_eng)

        if self.slip_middle:
            D_mtx = zeros((5, 5))
            sigma = zeros(5)
            if dbond[2] > 0.:
                D_mtx[2, 2] = 0.
                sigma[2] = self.T_fr
            else:
                D_mtx[2, 2] = self.bond_fn.get_diff(eps_app_eng[2])
                sigma[2] = self.bond_fn.get_value(eps_app_eng[2])  # tau_m
        else:
            D_mtx = zeros((4, 4))
            sigma = zeros(4)

        if dbond[0] > 0.:  # bond_l  
            D_mtx[0, 0] = 0.
            sigma[0] = self.T_fr
        else:
            D_mtx[0, 0] = self.bond_fn.get_diff(eps_app_eng[0])
            sigma[0] = self.bond_fn.get_value(eps_app_eng[0])  # tau_l

        if dbond[1] > 0.:  # bond_r  
            D_mtx[1, 1] = 0.
            sigma[1] = self.T_fr
        else:
            D_mtx[1, 1] = self.bond_fn.get_diff(eps_app_eng[1])
            sigma[1] = self.bond_fn.get_value(eps_app_eng[1])  # tau_r

        D_mtx[-2, -2] = self.Ef * self.Af
        D_mtx[-1, -1] = self.Em * self.Am

        sigma[-2] = self.Ef * self.Af * eps_app_eng[-2]  # sig_f
        sigma[-1] = self.Em * self.Am * eps_app_eng[-1]  # sig_m

        # You print the stress you just computed and the value of the apparent E

        return  sigma, D_mtx

    def _get_state_variables(self, sctx, eps_app_eng):
        dbond_l, dbond_r = sctx.mats_state_array[:2]

        if eps_app_eng[0] > self.s_cr:
            dbond_l = 1.
        if eps_app_eng[1] > self.s_cr:
            dbond_r = 1.

        if self.slip_middle:
            dbond_m = sctx.mats_state_array[-1]
            if eps_app_eng[2] > self.s_cr:
                dbond_m = 1.
            return dbond_l, dbond_r, dbond_m
        else:
            return dbond_l, dbond_r

    #---------------------------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #---------------------------------------------------------------------------------------------

    def get_sig_app(self, sctx, eps_app_eng, *args, **kw):
        # @TODO
        # the stress calculation is performed twice - it might be
        # cached.
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        r_pnt = sctx.loc
        s_tensor = zeros((3, 3))
        if r_pnt[1] >= 0.:
            s_tensor[0, 0] = sig_eng[-1]
        else:
            s_tensor[0, 0] = sig_eng[-2]
        return s_tensor

    def get_shear(self, sctx, eps_app_eng):
        # @TODO
        # the stress calculation is performed twice - it might be
        # cached.
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        r_pnt = sctx.loc
        if eps_app_eng.shape[0] == 4:
            if r_pnt[0] >= 0.:
                sig_pos = [sig_eng[1]]
            else:
                sig_pos = [sig_eng[0]]
        elif eps_app_eng.shape[0] == 5:
            if r_pnt[0] >= 0.5:
                sig_pos = [sig_eng[1]]
            elif r_pnt[0] <= -0.5:
                sig_pos = [sig_eng[0]]
            else:
                sig_pos = [sig_eng[2]]
        return sig_pos

    def get_eps_app(self, sctx, eps_app_eng):
        r_pnt = sctx.loc
        if r_pnt[1] >= 0.:
            eps_pos = [eps_app_eng[-1]]
        else:
            eps_pos = [eps_app_eng[-2]]
        return eps_pos

    def get_slip(self, sctx, eps_app_eng):
        r_pnt = sctx.loc
        if eps_app_eng.shape[0] == 4:
            if r_pnt[0] >= 0.:
                eps_pos = [eps_app_eng[1]]
            else:
                eps_pos = [eps_app_eng[0]]
        if eps_app_eng.shape[0] == 5:
            if r_pnt[0] >= 0.5:
                eps_pos = [eps_app_eng[1]]
            elif r_pnt[0] <= -0.5:
                eps_pos = [eps_app_eng[0]]
            else:
                eps_pos = [eps_app_eng[2]]
        return eps_pos

    def get_debonding(self, sctx, eps_app_eng):
        r_pnt = sctx.loc
        if self.slip_middle:
            if r_pnt[0] >= -0.25 and r_pnt[1] <= 0.25:
                dbond = sctx.mats_state_array[2]
            elif r_pnt[0] > 0.25:
                dbond = sctx.mats_state_array[1]
            else:
                dbond = sctx.mats_state_array[0]
        else:
            if r_pnt[0] >= 0.:
                dbond = sctx.mats_state_array[1]
            else:
                dbond = sctx.mats_state_array[0]
        return array([dbond], dtype=float_)

    #---------------------------------------------------------------------------------------------
    # Response trace evaluators
    #---------------------------------------------------------------------------------------------

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app_t1d'      : self.get_sig_app,
                'eps_app_t1d'      : self.get_eps_app,
                'shear_s'        : self.get_shear,
                'slip_s'         : self.get_slip,
                'debonding'      : self.get_debonding,
                'msig_pos'     : self.get_msig_pos,
                'msig_pm'      : self.get_msig_pm}


if __name__ == '__main__':
     mats_bond = MATS1D5Bond()
     mats_bond.configure_traits()
