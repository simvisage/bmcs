'''
Created on 12.01.2016
@author: Yingxiong, ABaktheer, RChudoba
'''

from bmcs.matmod.tloop import TLoop, TLine
from bmcs.matmod.tstepper import TStepper
from ibvpy.api import BCDof
from mathkit.mfn import MFnLineArray
from traits.api import \
    Property, Instance, cached_property, Str, Enum, \
    Range, on_trait_change, List, Float, Trait, Int
from traitsui.api import \
    View, Item, Group, VGroup, VSplit
from util.traits.editors import MPLFigureEditor
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSTreeNode, BMCSLeafNode
from view.window import BMCSWindow

from fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from mats_bondslip import MATSEvalFatigue
import numpy as np


# from ibvpy.api import TLoop, TLine, TStepper
class Material(BMCSLeafNode):

    node_name = Str('material parameters')
    E_b = Float(100,
                label="E_b ",
                desc="Bond Stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(10,
                  label="Gamma ",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(10,
              label="K ",
              desc="Isotropic harening",
              enter_set=True,
              auto_set=False)

    S = Float(0.01,
              label="S ",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    r = Float(1,
              label="r ",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    c = Float(1.2,
              label="c ",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    tau_pi_bar = Float(0.5,
                       label="Tau_pi_bar ",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    pressure = Float(-5,
                     label="Pressure",
                     desc="Lateral pressure",
                     enter_set=True,
                     auto_set=False)

    a = Float(1.7,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    view = View(VGroup(Group(Item('E_b'),
                             Item('tau_pi_bar'), show_border=True, label='Bond Stiffness and reversibility limit'),
                       Group(Item('gamma'),
                             Item('K'), show_border=True, label='Hardening parameters'),
                       Group(Item('S'),
                             Item('r'), Item('c'), show_border=True, label='Damage cumulation parameters'),
                       Group(Item('pressure'),
                             Item('a'), show_border=True, label='Lateral Pressure')))


class Geometry(BMCSLeafNode):

    node_name = Str('Geometry')
    L_x = Range(1, 700, value=300)
    A_m = Float(100 * 8 - 9 * 1.85, desc='matrix area [mm2]')
    A_f = Float(9 * 1.85, desc='reinforcement area [mm2]')
    P_b = Float(10., desc='perimeter of the bond interface [mm]')


class LoadingScenario(MFnLineArray, BMCSLeafNode):

    node_name = Str('Loading Scenario')
    number_of_cycles = Float(1.0,
                             input=True)
    maximum_loading = Float(5.0,
                            input=True)
    unloading_ratio = Range(0., 1., value=0.5,
                            input=True)
    number_of_increments = Float(10,
                                 input=True)
    loading_type = Enum("Monotonic", "Cyclic",
                        input=True)
    amplitude_type = Enum("Increased_Amplitude", "Constant_Amplitude",
                          input=True)
    loading_range = Enum("Non_symmetric", "Symmetric",
                         input=True)

    time = Range(0.00, 1.00, value=1.00)

    d_t = Float(0.05)
    t_max = Float(1.)
    k_max = Float(200)
    tolerance = Float(1e-4)

    def __init__(self, *arg, **kw):
        super(LoadingScenario, self).__init__(*arg, **kw)
        self._update_xy_arrays()

    @on_trait_change('+input')
    def _update_xy_arrays(self):
        print 'updating array'
        if self.loading_type == "Monotonic":
            self.number_of_cycles = 1
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 0] *= 0
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1],
                                           self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

            return d_arr

        if self.amplitude_type == "Increased_Amplitude" and self.loading_range == "Symmetric":
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] *= -1
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

            return d_arr

        if self.amplitude_type == "Increased_Amplitude" and self.loading_range == "Non_symmetric":
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] *= 0
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

            return d_arr

        if self.amplitude_type == "Constant_Amplitude" and self.loading_range == "Symmetric":
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] = -self.maximum_loading
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 1] = self.maximum_loading
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

            return d_arr

        if self.amplitude_type == "Constant_Amplitude" and self.loading_range == "Non_symmetric":
            # d_1 = np.zeros(self.number_of_cycles*2 + 1)
            d_1 = np.zeros(1)
            d_2 = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_2.reshape(-1, 2)[:, 0] = self.maximum_loading
            d_2.reshape(-1, 2)[:, 1] = self.maximum_loading * \
                self.unloading_ratio
            d_history = d_2.flatten()
            d_arr = np.hstack((d_1, d_history))

            return d_arr
        t_arr = np.linspace(0, self.t_max, len(self.d_array))
        self.xdata = t_arr
        self.ydata = self.d_array
        self.replot()

    traits_view = View(
        VGroup(
            VSplit(
                VGroup(
                    Group(
                        Item('loading_type')),
                    Group(
                        Item('maximum_loading')),
                    Group(
                        Item('number_of_cycles'),
                        Item('amplitude_type'),
                        Item('loading_range'),
                        Item('unloading_ratio'),
                        show_border=True, label='Cyclic load inputs'),
                    Group(
                        Item('d_t'),
                        Item('t_max'),
                        Item('k_max'), show_border=True, label='Solver Settings')
                ),
                Item('figure', editor=MPLFigureEditor(),
                     height=300,
                     show_label=False,
                     resizable=True,
                     springy=True),
            )
        )
    )

    tree_view = traits_view


class Viz2DPullOutFW(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    def plot(self, ax, vot, *args, **kw):
        P_t = self.vis2d.get_P_t()
        w_t = self.vis2d.get_w_t()
        ax.plot(w_t, P_t, *args, **kw)


class Viz2DPullOutField(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = Property(depends_on='plot_fn')

    @cached_property
    def _get_label(self):
        return 'field: %s' % self.plot_fn

    plot_fn = Trait('u_C',
                    {'eps_C': 'plot_eps_C',
                     'u_C': 'plot_u_C',
                     's': 'plot_s',
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


class PullOutSimulation(BMCSTreeNode, Vis2D):

    node_name = Str('pull out simulation')

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [self.material,
                self.geometry,
                self.loading_scenario,
                ]

    material = Instance(Material)

    def _material_default(self):
        return Material()

    loading_scenario = Instance(LoadingScenario)

    def _loading_scenario_default(self):
        return LoadingScenario()

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    controlled_dof = Int(-1)
    mats_eval = Property(Instance(MATSEvalFatigue))
    '''Material model'''
    @cached_property
    def _get_mats_eval(self):
        # assign the material parameters
        return MATSEvalFatigue(E_b=self.material.E_b,
                               gamma=self.material.gamma,
                               S=self.material.S,
                               tau_pi_bar=self.material.tau_pi_bar,
                               r=self.material.r,
                               K=self.material.K,
                               c=self.material.c,
                               a=self.material.a,
                               pressure=self.material.pressure)

    fets_eval = Property(Instance(FETS1D52ULRHFatigue))
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        # assign the geometry parameters
        return FETS1D52ULRHFatigue(A_m=self.geometry.A_m,
                                   P_b=self.geometry.P_b,
                                   A_f=self.geometry.A_f)

    tstepper = Property(Instance(TStepper))
    '''Objects representing the state of the model providing
    the predictor and corrector functionality needed for time-stepping
    algorithm.
    '''
    @cached_property
    def _get_tstepper(self):
        bc_list = [BCDof(node_name='fixed left end', var='u',
                         dof=0, value=0.0),
                   BCDof(node_name='pull-out displacement', var='u',
                         dof=self.controlled_dof, value=200.0,
                         time_function=self.loading_scenario.time_func)]

        return TStepper(node_name='Pull-out',
                        L_x=self.geometry.L_x,
                        bcond_list=bc_list
                        )

    tloop = Property(Instance(TLoop))
    '''Algorithm controlling the time stepping.
    '''
    @cached_property
    def _get_tloop(self):

        # assign the parameters for solver and loading_scenario
        t_max = self.loading_scenario.t_max
        d_t = self.loading_scenario.d_t
        k_max = self.loading_scenario.k_max
        tolerance = self.loading_scenario.tolerance

        tline = TLine(min=0.0, step=d_t, max=t_max,
                      time_change_notifier=self.time_changed,
                      )
        return TLoop(ts=self.tstepper, k_max=k_max,
                     tolerance=tolerance,
                     tline=tline)

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

    def get_s(self, vot):
        '''Slip between the two material phases'''
        d_ECid = self.get_d_ECid(vot)
        s_Emd = np.einsum('Cim,ECid->Emd', self.tstepper.sN_Cim, d_ECid)
        return s_Emd.flatten()

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

    def plot_s(self, ax, vot):
        ax.plot(self.tstepper.X_J, self.get_s(vot))

    def plot_eps_C(self, ax, vot):
        ax.plot(self.tstepper.X_M, self.get_eps_C(vot))

    def plot_w(self, ax, vot):
        ax.plot(self.tstepper.X_J, self.get_w(vot))

    trait_view = View(Item('fets_eval'),
                      )

    viz2d_classes = {'field': Viz2DPullOutField,
                     'F-w': Viz2DPullOutFW
                     }


if __name__ == '__main__':
    po = PullOutSimulation()
    po.loading_scenario.configure_traits()
#    po.tloop.eval()
    w = BMCSWindow(model=po)
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'w', plot_fn='w')
    po.add_viz2d('F-w')
    w.configure_traits()
