'''
Created on 12.01.2016
@author: Yingxiong, ABaktheer, RChudoba
'''

from ibvpy.api import BCDof
from ibvpy.core.bcond_mngr import BCondMngr
from mathkit.mfn import MFnLineArray
from traits.api import \
    Property, Instance, cached_property, Str, Enum, \
    Range, on_trait_change, List, Float, Trait, Int
from traitsui.api import \
    View, UItem, Item, Group, VGroup, VSplit
from util.traits.editors import MPLFigureEditor
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSModel, BMCSWindow, TLine

from bmcs.mats.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from bmcs.mats.mats_bondslip import MATSEvalFatigue
from bmcs.mats.tloop import TLoop
from bmcs.mats.tstepper import TStepper
import numpy as np


class Material(BMCSLeafNode):

    node_name = Str('material parameters')
    E_b = Float(12900,
                MAT=True,
                input=True,
                label="E_b ",
                desc="Bond Stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(60,
                  MAT=True,
                  input=True,
                  label="Gamma ",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(10,
              MAT=True,
              input=True,
              label="K ",
              desc="Isotropic harening",
              enter_set=True,
              auto_set=False)

    S = Float(0.001,
              MAT=True,
              input=True,
              label="S ",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    r = Float(0.7,
              MAT=True,
              input=True,
              label="r ",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    c = Float(1.5,
              MAT=True,
              input=True,
              label="c ",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    tau_pi_bar = Float(4.5,
                       MAT=True,
                       input=True,
                       label="Tau_pi_bar ",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    pressure = Float(0,
                     MAT=True,
                     input=True,
                     label="Pressure",
                     desc="Lateral pressure",
                     enter_set=True,
                     auto_set=False)

    a = Float(1.7,
              MAT=True,
              input=True,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    view = View(VGroup(Group(Item('E_b'),
                             Item('tau_pi_bar'), show_border=True,
                             label='Bond Stiffness and reversibility limit'),
                       Group(Item('gamma'),
                             Item('K'), show_border=True,
                             label='Hardening parameters'),
                       Group(Item('S'),
                             Item('r'), Item('c'), show_border=True,
                             label='Damage cumulation parameters'),
                       Group(Item('pressure'),
                             Item('a'), show_border=True,
                             label='Lateral Pressure')))

    tree_view = view


class Geometry(BMCSLeafNode):

    node_name = 'geometry'
    L_x = Float(45,
                GEO=True,
                input=True,
                auto_set=False, enter_set=True,
                desc='embedded length')
    A_m = Float(15240,
                CS=True,
                input=True,
                auto_set=False, enter_set=True,
                desc='matrix area [mm2]')
    A_f = Float(153.9,
                CS=True,
                input=True,
                auto_set=False, enter_set=True,
                desc='reinforcement area [mm2]')
    P_b = Float(44,
                input=True,
                auto_set=False, enter_set=True,
                desc='perimeter of the bond interface [mm]')

    view = View(
        Item('L_x'),
        Item('A_m'),
        Item('A_f'),
        Item('P_b')
    )

    tree_view = view


class LoadingScenario(MFnLineArray, BMCSLeafNode):

    node_name = Str('Loading Scenario')
    number_of_cycles = Int(1,
                           BC=True,
                           input=True)
    maximum_loading = Float(1.0,
                            BC=True,
                            input=True)
    unloading_ratio = Range(0., 1., value=0.5,
                            BC=True,
                            input=True)
    number_of_increments = Int(10,
                               BC=True,
                               input=True)
    loading_type = Enum("Monotonic", "Cyclic",
                        BC=True,
                        input=True)
    amplitude_type = Enum("Increased_Amplitude", "Constant_Amplitude",
                          BC=True,
                          input=True)
    loading_range = Enum("Non_symmetric", "Symmetric",
                         BC=True,
                         input=True)

    t_max = Float(1.)

    def __init__(self, *arg, **kw):
        super(LoadingScenario, self).__init__(*arg, **kw)
        self._update_xy_arrays()

    @on_trait_change('+BC')
    def _update_xy_arrays(self):
        if (self.loading_type == "Monotonic"):
            self.number_of_cycles = 1
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 0] *= 0
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1],
                                           self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if (self.amplitude_type == "Increased_Amplitude" and
                self.loading_range == "Symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] *= -1
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1],
                                           self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if (self.amplitude_type == "Increased_Amplitude" and
                self.loading_range == "Non_symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] *= 0
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1],
                                           self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if (self.amplitude_type == "Constant_Amplitude" and
                self.loading_range == "Symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] = -self.maximum_loading
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 1] = self.maximum_loading
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if (self.amplitude_type == "Constant_Amplitude" and
                self.loading_range == "Non_symmetric"):
            # d_1 = np.zeros(self.number_of_cycles*2 + 1)
            d_1 = np.zeros(1)
            d_2 = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_2.reshape(-1, 2)[:, 0] = self.maximum_loading
            d_2.reshape(-1, 2)[:, 1] = self.maximum_loading * \
                self.unloading_ratio
            d_history = d_2.flatten()
            d_arr = np.hstack((d_1, d_history))

        t_arr = np.linspace(0, self.t_max, len(d_arr))
        self.xdata = t_arr
        self.ydata = d_arr
        self.replot()

    traits_view = View(
        VGroup(
            VSplit(
                VGroup(
                    Group(
                        Item('loading_type',
                             full_size=True, resizable=True
                             )
                    ),
                    Group(
                        Item('maximum_loading',
                             full_size=True, resizable=True)
                    ),
                    Group(
                        Item('number_of_cycles',
                             full_size=True, resizable=True),
                        Item('amplitude_type'),
                        Item('loading_range'),
                        Item('unloading_ratio'),
                        show_border=True, label='Cyclic load inputs'),
                    scrollable=True
                ),
                UItem('figure', editor=MPLFigureEditor(),
                      height=300,
                      resizable=True,
                      springy=True),
            )
        )
    )

    tree_view = traits_view


class Viz2DLoadControlFunction(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'Load control'

    def plot(self, ax, vot, *args, **kw):
        bc = self.vis2d.control_bc
        val = bc.value
        tloop = self.vis2d.tloop
        t_arr = np.array(tloop.t_record, np.float_)
        f_arr = val * bc.time_function(t_arr)
        ax.plot(t_arr, f_arr, 'bo-')
        vot_idx = tloop.get_time_idx(vot)
        ax.plot([t_arr[vot_idx]], [f_arr[vot_idx]], 'ro')
        var = bc.var


class Viz2DPullOutFW(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    def plot(self, ax, vot, *args, **kw):
        idx = self.vis2d.tloop.get_time_idx(vot)
        P_t = self.vis2d.get_P_t()
        w_t = self.vis2d.get_w_t()
        ax.plot(w_t, P_t, *args, **kw)
        P, w = P_t[idx], w_t[idx]
        ax.plot([w], [P], 'ro')


class Viz2DPullOutField(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = Property(depends_on='plot_fn')

    @cached_property
    def _get_label(self):
        return 'field: %s' % self.plot_fn

    plot_fn = Trait('u_C',
                    {'eps_C': 'plot_eps_C',
                     'sig_C': 'plot_sig_C',
                     'u_C': 'plot_u_C',
                     's': 'plot_s',
                     'sf': 'plot_sf',
                     'w': 'plot_w',
                     'Fint_C': 'plot_Fint_C'
                     },
                    label='Field',
                    tooltip='Select the field to plot'
                    )

    def plot(self, ax, vot, *args, **kw):
        getattr(self.vis2d, self.plot_fn_)(ax, vot, *args, **kw)
        ymin, ymax = ax.get_ylim()
        self.y_max = max(ymax, self.y_max)
        self.y_min = min(ymin, self.y_min)
        ax.set_ylim(ymin=self.y_min, ymax=self.y_max)

    y_max = Float(1.0, label='Y-max value',
                  auto_set=False, enter_set=True)
    y_min = Float(0.0, label='Y-min value',
                  auto_set=False, enter_set=True)

    traits_view = View(
        Item('plot_fn'),
        Item('y_min', ),
        Item('y_max', )
    )


class PullOutSimulation(BMCSModel, Vis2D):

    node_name = 'pull out simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.material,
            self.geometry,
            self.bcond_mngr,
        ]

    def eval(self):
        return self.tloop.eval()

    def paused(self):
        self.tloop.paused = True

    def stop(self):
        self.tloop.restart = True

    material = Instance(Material)

    def _material_default(self):
        return Material()

    loading_scenario = Instance(LoadingScenario)

    def _loading_scenario_default(self):
        return LoadingScenario()

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    n_e_x = Int(20, auto_set=False, enter_set=True)

    w_max = Float(1, auto_set=False, enter_set=True)

    controlled_dof = Property

    def _get_controlled_dof(self):
        return 2 + 2 * self.n_e_x - 1

    mats_eval = Property(Instance(MATSEvalFatigue),
                         depends_on='MAT')
    '''Material model'''
    @cached_property
    def _get_mats_eval(self):
        # assign the material parameters
        print 'new material model'
        return MATSEvalFatigue(E_b=self.material.E_b,
                               gamma=self.material.gamma,
                               S=self.material.S,
                               tau_pi_bar=self.material.tau_pi_bar,
                               r=self.material.r,
                               K=self.material.K,
                               c=self.material.c,
                               a=self.material.a,
                               pressure=self.material.pressure)

    fets_eval = Property(Instance(FETS1D52ULRHFatigue),
                         depends_on='CS')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        # assign the geometry parameters
        return FETS1D52ULRHFatigue(A_m=self.geometry.A_m,
                                   P_b=self.geometry.P_b,
                                   A_f=self.geometry.A_f)

    bcond_mngr = Instance(BCondMngr)
    '''Boundary condition manager
    '''

    def _bcond_mngr_default(self):
        bc_list = [BCDof(node_name='fixed left end', var='u',
                         dof=0, value=0.0),
                   BCDof(node_name='pull-out displacement', var='u',
                         dof=self.controlled_dof, value=self.w_max,
                         time_function=self.loading_scenario)]
        return BCondMngr(bcond_list=bc_list)

    control_bc = Property(depends_on='BC')
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return self.bcond_mngr.bcond_list[1]

    tstepper = Property(Instance(TStepper),
                        depends_on='MAT,GEO,MESH,CS,TIME,ALG,BC')
    '''Objects representing the state of the model providing
    the predictor and corrector functionality needed for time-stepping
    algorithm.
    '''
    @cached_property
    def _get_tstepper(self):
        return TStepper(node_name='Pull-out',
                        n_e_x=self.n_e_x,
                        mats_eval=self.mats_eval,
                        fets_eval=self.fets_eval,
                        L_x=self.geometry.L_x,
                        bcond_mngr=self.bcond_mngr
                        )

    tline = Instance(TLine)

    def _tline_default(self):
        # assign the parameters for solver and loading_scenario
        t_max = 1.0  # self.loading_scenario.t_max
        d_t = 0.02  # self.loading_scenario.d_t
        return TLine(min=0.0, step=d_t, max=t_max,
                     time_change_notifier=self.time_changed,
                     )

    k_max = Int(200,
                ALG=True)
    tolerance = Float(1e-4,
                      ALG=True)

    tloop = Property(Instance(TLoop),
                     depends_on='MAT,GEO,MESH,CS,TIME,ALG,BC')
    '''Algorithm controlling the time stepping.
    '''
    @cached_property
    def _get_tloop(self):

        k_max = self.k_max
        tolerance = self.tolerance
        print 'NEW TLOOP'
        return TLoop(ts=self.tstepper, k_max=k_max,
                     tolerance=tolerance,
                     tline=self.tline)

    def time_changed(self, time):
        if self.ui != None:
            self.ui.viz_sheet.time_changed(time)

    def time_range_changed(self, tmax):
        self.tline.max = tmax
        self.ui.viz_sheet.time_range_changed(tmax)

    def set_tmax(self, time):
        self.time_range_changed(time)

    def get_d_ECid(self, vot):
        '''Get the displacements as a four-dimensional array 
        corresponding to element, material component, node, spatial dimension
        '''
        idx = self.tloop.get_time_idx(vot)
        d = self.tloop.U_record[idx]
        dof_ECid = self.tstepper.dof_ECid
        return d[dof_ECid]

    def get_u_C(self, vot):
        '''Displacement field
        '''
        d_ECid = self.get_d_ECid(vot)
        N_mi = self.fets_eval.N_mi
        u_EmdC = np.einsum('mi,ECid->EmdC', N_mi, d_ECid)
        return u_EmdC.reshape(-1, 2)

    def get_eps_C(self, vot):
        '''Epsilon in the components'''
        d_ECid = self.get_d_ECid(vot)
        eps_EmdC = np.einsum('Eimd,ECid->EmdC', self.tstepper.dN_Eimd, d_ECid)
        return eps_EmdC.reshape(-1, 2)

    def get_sig_C(self, vot):
        '''Get streses in the components 
        @todo: unify the index protocol
        for eps and sig. Currently eps uses multi-layer indexing, sig
        is extracted from the material model format.
        '''
        idx = self.tloop.get_time_idx(vot)
        return self.tloop.sig_EmC_record[idx].reshape(-1, 2)

    def get_s(self, vot):
        '''Slip between the two material phases'''
        d_ECid = self.get_d_ECid(vot)
        s_Emd = np.einsum('Cim,ECid->Emd', self.tstepper.sN_Cim, d_ECid)
        return s_Emd.flatten()

    def get_sf(self, vot):
        '''Get the shear flow in the interface
        @todo: unify the index protocol
        for eps and sig. Currently eps uses multi-layer indexing, sig
        is extracted from the material model format.
        '''
        idx = self.tloop.get_time_idx(vot)
        sf = self.tloop.sf_Em_record[idx].flatten()
        return sf

    def get_P_t(self):
        F_array = np.array(self.tloop.F_record, dtype=np.float_)
        return F_array[:, self.controlled_dof]

    def get_w_t(self):
        U_array = np.array(self.tloop.U_record, dtype=np.float_)
        return U_array[:, self.controlled_dof]

    def get_w(self, vot):
        '''Damage variables
        '''
        idx = self.tloop.get_time_idx(vot)
        w_Emd = self.tloop.w_record[idx]
        return w_Emd.flatten()

    def plot_u_C(self, ax, vot):
        ax.plot(self.tstepper.X_J, self.get_u_C(vot))

    def plot_eps_C(self, ax, vot):
        ax.plot(self.tstepper.X_M, self.get_eps_C(vot))

    def plot_sig_C(self, ax, vot):
        ax.plot(self.tstepper.X_M, self.get_sig_C(vot))

    def plot_s(self, ax, vot):
        ax.plot(self.tstepper.X_J, self.get_s(vot))

    def plot_sf(self, ax, vot):
        ax.plot(self.tstepper.X_J, self.get_sf(vot))

    def plot_w(self, ax, vot):
        ax.plot(self.tstepper.X_J, self.get_w(vot))

    trait_view = View(Item('fets_eval'),
                      )

    viz2d_classes = {'field': Viz2DPullOutField,
                     'F-w': Viz2DPullOutFW,
                     'load function': Viz2DLoadControlFunction,
                     }


def run_debontrix_cumslide():
    po = PullOutSimulation(n_e_x=100, k_max=500)
#     po.geometry.set(L_x=450)
    po.tline.step = 0.05
    po.bcond_mngr.bcond_list[1].value = 0.01
#     print po.tloop
#     po.material.set(tau_pi_bar=1)
#     print po.tloop
    po.tloop.eval()

    w = BMCSWindow(model=po,
                   offline=True)
    po.add_viz2d('load function')
    po.add_viz2d('F-w')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'w', plot_fn='w')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    w.set(offline=False)
    po.add_viz2d('field', 'sf', plot_fn='sf')
#     po.material.set(tau_pi_bar=1)
    w.configure_traits()

if __name__ == '__main__':
    run_debontrix_cumslide()