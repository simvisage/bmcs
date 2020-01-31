
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
mpl_matplotlib_editor = MFnMatplotlibEditor()

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
    G = Float(1.,  # 34e+3,
                 label="G",
                 desc="Shear Stiffness",
                 auto_set=False, enter_set=True)
    bond_fn = Trait(MFnLineArray(ydata=[0, 1]),
                     label="Bond",
                     desc="Bond Function",
                     auto_set=False)

    traits_view = View(Item('Ef'),
                        Item('Af'),
                        Item('Em'),
                        Item('Am'),
                        Item('G'),
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
        '''
        if eps_app_eng.shape[0] == 4:
            D_mtx = zeros((4, 4))
            sigma = zeros(4)
        elif eps_app_eng.shape[0] == 5:
            D_mtx = zeros((5, 5))
            sigma = zeros(5)
            D_mtx[2, 2] = self.bond_fn.get_diff(eps_app_eng[2])
            sigma[2] = self.bond_fn.get_value(eps_app_eng[2])  # tau_m
        else:
            print("MATS1D5Bond: Unsupported number of strain components")

        D_mtx[0, 0] = self.bond_fn.get_diff(eps_app_eng[0])
        D_mtx[1, 1] = self.bond_fn.get_diff(eps_app_eng[1])

        D_mtx[-2, -2] = self.Ef * self.Af
        D_mtx[-1, -1] = self.Em * self.Am

        sigma[0] = self.bond_fn.get_value(eps_app_eng[0])  # tau_l
        sigma[1] = self.bond_fn.get_value(eps_app_eng[1])  # tau_r

        sigma[-2] = self.Ef * self.Af * eps_app_eng[-2]  # sig_f
        sigma[-1] = self.Em * self.Am * eps_app_eng[-1]  # sig_m

        # You print the stress you just computed and the value of the apparent E

        return  sigma, D_mtx

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
                'msig_pos'     : self.get_msig_pos,
                'msig_pm'      : self.get_msig_pm}


if __name__ == '__main__':
     mats_bond = MATS1D5Bond()
     mats_bond.configure_traits()
