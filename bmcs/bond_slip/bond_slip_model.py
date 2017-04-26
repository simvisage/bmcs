'''
Created on 12.12.2016

@author: abaktheer, rch

@todo: reset loading scenario upon restart
@todo: issue restart upon a change of a material parameter
@todo: more natural representation of polymorphic tree nodes
@todo: loading_scenario - two classes of attributes - categoric and numeric
        check the dependencies between them, so that they can be set.
'''


from bmcs.pullout.pullout import LoadingScenario
from ibvpy.api import IMATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import \
    Property, Instance, cached_property, Str, \
    List, Float, Trait, on_trait_change, Bool, Dict,\
    HasStrictTraits, Int, Tuple, Interface, implements
from traitsui.api import \
    View, Item, UItem, Group, VGroup, VSplit
from traitsui.editors.enum_editor import EnumEditor
from view.examples.tfun_pwl_interactive import TFunPWLInteractive
from view.plot2d import Vis2D, Viz2D
from view.ui import BMCSLeafNode, BMCSTreeNode
from view.window.bmcs_window import BMCSModel, BMCSWindow

from mats_bondslip import MATSBondSlipD, MATSBondSlipDP, MATSBondSlipEP
import numpy as np


class PlottableFn(HasStrictTraits):

    plot_min = Float(0.0, input=True,
                     enter_set=True, auto_set=False)
    plot_max = Float(1.0, input=True,
                     enter_set=True, auto_set=False)

    fn = Instance(MFnLineArray)

    def _fn_default(self):
        return MFnLineArray()

    def __init__(self, *args, **kw):
        super(PlottableFn, self).__init__(*args, **kw)
        self.update()

    @on_trait_change('+input')
    def update(self):
        n_vals = 200
        xdata = np.linspace(self.plot_min, self.plot_max, n_vals)
        ydata = self.__call__(xdata)
        self.fn.set(xdata=xdata, ydata=ydata)
        self.fn.replot()

    traits_view = View(UItem('fn'))


class IDamageFn(Interface):
    pass


class JirasekDamageFn(BMCSLeafNode, PlottableFn):

    node_name = 'Jirasek damage function'

    implements(IDamageFn)

    s_0 = Float(5. / 12900.,
                MAT=True,
                input=True,
                label="s_0",
                desc="parameter controls the damage function",
                enter_set=True,
                auto_set=False)

    s_f = Float(0.001,
                MAT=True,
                input=True,
                label="s_f",
                desc="parameter controls the damage function",
                enter_set=True,
                auto_set=False)

    plot_max = 1e-2

    def __call__(self, kappa):
        s_0 = self.s_0
        s_f = self.s_f
        omega = np.zeros_like(kappa, dtype=np.float_)
        d_idx = np.where(kappa >= s_0)[0]
        k = kappa[d_idx]
        omega[d_idx] = 1. - s_0 / k * np.exp(-1 * (k - s_0) / s_f)
        return omega

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0', full_size=True, resizable=True),
                Item('s_f'),
                Item('plot_max'),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


class LiDamageFn(BMCSLeafNode, PlottableFn):

    node_name = 'Li damage function'

    implements(IDamageFn)

    s_0 = Float(0.00001,
                MAT=True,
                input=True,
                label="s_0",
                desc="parameter controls the damage function",
                enter_set=True,
                auto_set=False)

    alpha_1 = Float(1.,
                    MAT=True,
                    input=True,
                    label="alpha_1",
                    desc="parameter controls the damage function",
                    enter_set=True,
                    auto_set=False)

    alpha_2 = Float(1000.,
                    MAT=True,
                    input=True,
                    label="alpha_2",
                    desc="parameter controls the damage function",
                    enter_set=True,
                    auto_set=False)

    plot_max = 1e-2

    def __call__(self, kappa):
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        s_0 = self.s_0
        omega = np.zeros_like(kappa, dtype=np.float_)
        d_idx = np.where(kappa >= s_0)[0]
        k = kappa[d_idx]
        omega[d_idx] = 1. / (1. + np.exp(-1. * alpha_2 * k + 6.)) * alpha_1
        return omega

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0', full_size=True, resizable=True),
                Item('alpha_1'),
                Item('alpha_2'),
                Item('plot_max'),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


class AbaqusDamageFn(BMCSLeafNode, PlottableFn):

    node_name = 'Abaqus damage function'

    implements(IDamageFn)

    s_0 = Float(0.0004,
                MAT=True,
                input=True,
                label="s_0",
                desc="parameter controls the damage function",
                enter_set=True,
                auto_set=False)

    s_u = Float(0.003,
                MAT=True,
                input=True,
                label="s_u",
                desc="parameter controls the damage function",
                enter_set=True,
                auto_set=False)

    alpha = Float(0.1,
                  MAT=True,
                  input=True,
                  label="alpha",
                  desc="parameter controlling the slop of damage",
                  enter_set=True,
                  auto_set=False)

    plot_max = 1e-3

    def __call__(self, kappa):
        s_0 = self.s_0
        s_u = self.s_u
        alpha = self.alpha

        omega = np.zeros_like(kappa, dtype=np.float_)
        d_idx = np.where(kappa >= s_0)[0]
        k = kappa[d_idx]

        sk = (k - s_0) / (s_u - s_0)
        frac = (1 - np.exp(-alpha * sk)) / (1 - np.exp(-alpha))

        omega[d_idx] = 1 - s_0 / k * (1 - frac)
        omega[np.where(omega > 1.0)] = 1.0
        return omega

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0', full_size=True, resizable=True),
                Item('s_u'),
                Item('alpha'),
                Item('plot_max'),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


class Material(BMCSTreeNode):

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.omega_fn, ]

    @on_trait_change('omega_fn_type')
    def _update_node_list(self):
        self.tree_node_list = [self.omega_fn]

    node_name = 'material parameters'

    E_b = Float(12900.0,
                MAT=True,
                label="G",
                desc="Shear Stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(0,
                  MAT=True,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(0,
              MAT=True,
              label="K",
              desc="Isotropic harening",
              enter_set=True,
              auto_set=False)

    tau_bar = Float(5.0,
                    MAT=True,
                    label="Tau_0",
                    desc="yield stress",
                    enter_set=True,
                    auto_set=False)

    def __init__(self, *args, **kw):
        super(Material, self).__init__(*args, **kw)
        self._update_s0()

    @on_trait_change('tau_bar,E_b')
    def _update_s0(self):
        s_0 = self.tau_bar / self.E_b
        self.omega_fn.s_0 = s_0

    omega_fn_type = Trait('li',
                          dict(li=LiDamageFn,
                               jirasek=JirasekDamageFn,
                               abaqus=AbaqusDamageFn),
                          MAT=True,
                          )

    @on_trait_change('omega_fn_type')
    def _reset_omega_fn(self):
        self.omega_fn = self.omega_fn_type_()

    omega_fn = Instance(IDamageFn,
                        MAT=True)

    def _omega_fn_default(self):
        return LiDamageFn()

    view = View(
        Group(
            VSplit(
                VGroup(
                    Group(
                        Item('E_b', full_size=True, resizable=True),
                        Item('tau_bar'),
                        show_border=True,
                        label='Bond Stiffness and yield stress'
                    ),
                    Group(
                        Item('gamma', full_size=True, resizable=True),
                        Item('K'),
                        show_border=True,
                        label='Hardening parameters'
                    ),
                ),
                Item('omega_fn_type', full_size=True, resizable=True),
                Group(
                    UItem('omega_fn@', full_size=True, resizable=True),
                    show_border=True,
                    label='Damage function'
                ),
            )
        )
    )
    tree_view = view


class Viz2DBondHistory(Viz2D):

    def __init__(self, *args, **kw):
        super(Viz2DBondHistory, self).__init__(*args, **kw)
        self.on_trait_change(self._set_sv_names, 'vis2d.material_model')
        self._set_sv_names()

    x_sv_name = Str
    y_sv_name = Str
    sv_names = List([])

    def _set_sv_names(self):
        self.sv_names = self.vis2d.sv_names

    def plot(self, ax, vot, *args, **kw):

        xdata = self.vis2d.get_sv_hist(self.x_sv_name)
        ydata = self.vis2d.get_sv_hist(self.y_sv_name)
        ax.plot(xdata, ydata)
        ax.set_xlabel(self.x_sv_name)
        ax.set_ylabel(self.y_sv_name)
        # ax.legend()

    traits_view = View(
        Item('x_sv_name', editor=EnumEditor(name='sv_names')),
        Item('y_sv_name', editor=EnumEditor(name='sv_names'))
    )


class BondSlipModel(BMCSModel, Vis2D):

    node_name = Str('Bond slip model')

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.tline,
                self.material,
                self.loading_scenario]

    @on_trait_change('MAT,+BC')
    def _update_node_list(self):
        self.tree_node_list = [self.tline,
                               self.material,
                               self.loading_scenario]

    material = Instance(Material)
    '''Record of material parameters that are accessed by the
    individual models.
    '''

    def _material_default(self):
        return Material()

    interaction_type = Trait('interactive',
                             {'interactive': TFunPWLInteractive,
                              'predefined': LoadingScenario},
                             BC=True,
                             )
    '''Type of control - either interactive or predefined.
    '''

    def _interaction_type_changed(self):
        print 'assigning interaction type'
        self.loading_scenario = self.interaction_type_()

    loading_scenario = Instance(MFnLineArray)
    '''Loading scenario in form of a function that maps the time variable
    to the control slip.
    '''

    def _loading_scenario_default(self):
        return TFunPWLInteractive()

    material_model = Trait('plasticity',
                           {'damage': MATSBondSlipD,
                            'plasticity': MATSBondSlipEP,
                            'damage-plasticity': MATSBondSlipDP,
                            },
                           enter_set=True, auto_set=False,
                           MAT=True
                           )
    '''Available material models.
    '''

    mats_eval = Property(Instance(IMATSEval), depends_on='MAT')
    '''Instance of the material model.
    '''
    @cached_property
    def _get_mats_eval(self):
        return self.material_model_(material=self.material)

    sv_names = Property(List(Str), depends_on='material_model')
    '''Names of state variables of the current material model.
    '''
    @cached_property
    def _get_sv_names(self):
        return ['t', 's'] + self.mats_eval.sv_names

    sv_hist = Dict((Str, List))
    '''Record of state variables. The number of names of the variables
    depend on the active material model. See s_names.
    '''

    def _sv_hist_default(self):
        sv_hist = {}
        for sv_name in self.sv_names:
            sv_hist[sv_name] = []
        return sv_hist

    @on_trait_change('MAT,BC')
    def _sv_hist_reset(self):
        print 'sv_hist_reset'
        for sv_name in self.sv_names:
            self.sv_hist[sv_name] = []

    def paused(self):
        self._paused = True

    def stop(self):
        self._sv_hist_reset()
        self._restart = True
        self.loading_scenario.reset()

    _paused = Bool(False)
    _restart = Bool(True)

    @on_trait_change('MAT,BC,+BC')
    def signal_mat_changed(self):
        # @todo: review this - this sends a signal to the ui window
        # that the currentcalculation needs to be abandoned, since
        # continuation is not possible. The ui then initiates the reset
        # of the model state and of the loading history.
        if self.ui:
            self.ui.stop()

    n_steps = Int(1000, ALG=True,
                  enter_set=True, auto_set=False)

    state_vars = Tuple

    def init(self):
        if self._paused:
            self._paused = False
        if self._restart:
            self.tline.val = self.tline.min
            self.state_vars = self.mats_eval.init_state_vars()
            self._restart = False

    def eval(self):
        '''this method is just called by the tloop_thread'''

        t_max = self.loading_scenario.xdata[-1]
        print 'EVAL', t_max
        self.set_tmax(t_max)
        t_min = self.tline.val
        n_steps = self.n_steps
        tarray = np.linspace(t_min, t_max, n_steps)
        sv_names = self.sv_names
        sv_records = [[] for s_n in sv_names]

        s_last = 0
        for idx, t in enumerate(tarray):
            if self._restart or self._paused:
                break
            s = self.loading_scenario(t)
            d_s = s - s_last
            self.state_vars = \
                self.mats_eval.get_next_state(s, d_s, self.state_vars)
            # record the current simulation step
            for sv_record, state in zip(sv_records,
                                        (t, s) + self.state_vars):
                sv_record.append(np.copy(state))

        # append the data to the previous simulation steps
        for sv_name, sv_record in zip(sv_names, sv_records):
            self.sv_hist[sv_name].append(np.array(sv_record))

        self.tline.val = min(t, self.tline.max)

    def get_sv_hist(self, sv_name):
        return np.vstack(self.sv_hist[sv_name])

    viz2d_classes = {'bond history': Viz2DBondHistory}

    tree_view = View(Item('material_model'),
                     Item('interaction_type'))


def run_bond_slip_model_dp():
    bsm = BondSlipModel(interaction_type='predefined',
                        material_model='damage-plasticity')
    w = BMCSWindow(model=bsm, n_cols=1)
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 'omega-s', x_sv_name='s', y_sv_name='omega')
    bsm.add_viz2d('bond history', 's_p-s', x_sv_name='s', y_sv_name='s_p')
    bsm.loading_scenario.set(maximum_loading=0.01)
    w.configure_traits()


def run_bond_slip_model_d():
    bsm = BondSlipModel(interaction_type='predefined',
                        material_model='damage')
    w = BMCSWindow(model=bsm, n_cols=1)
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 'omega-s', x_sv_name='s', y_sv_name='omega')
    bsm.loading_scenario.set(loading_type='cyclic',
                             amplitude_type='constant'
                             )
    bsm.loading_scenario.set(number_of_cycles=1,
                             maximum_loading=0.01,
                             unloading_ratio=0.5)
    bsm.run()
    w.configure_traits()


def run_bond_slip_model_p():
    bsm = BondSlipModel(interaction_type='predefined',
                        material_model='plasticity',
                        n_steps=2000)
    w = BMCSWindow(model=bsm, n_cols=2)
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 's_p-s', x_sv_name='s', y_sv_name='s_p')
    bsm.add_viz2d('bond history', 'z-s', x_sv_name='s', y_sv_name='z')
    bsm.add_viz2d('bond history', 'alpha-s', x_sv_name='s', y_sv_name='alpha')
    bsm.loading_scenario.set(loading_type='cyclic',
                             amplitude_type='constant'
                             )
    bsm.loading_scenario.set(number_of_cycles=3,
                             maximum_loading=0.004,
                             unloading_ratio=0)
    bsm.material.set(gamma=101, K=-150)
    bsm.run()
    w.configure_traits()


def run_interactive_test():
    bsm = BondSlipModel(interaction_type='interactive')
    print 'set f_val'
    bsm.loading_scenario.f_value = 0.1
    print 'eval'
    bsm.eval()
    print 'set f_val'
    bsm.loading_scenario.f_value = 0.4
    print 'eval'
    bsm.eval()
    print bsm.sv_hist.keys()
    print bsm.get_sv_hist('omega')


def run_interactive_test_d():
    bsm = BondSlipModel(interaction_type='interactive',
                        material_model='damage',
                        n_steps=10)
    w = BMCSWindow(model=bsm, n_cols=1)
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 'omega-s', x_sv_name='s', y_sv_name='omega')
    bsm.loading_scenario.f_max = 0.005
    bsm.loading_scenario.f_value = 0.004
    print '04'
    bsm.loading_scenario.f_value = 0.001
    print '01'
    bsm.loading_scenario.f_value = 0.0045
    print '02'
    bsm.loading_scenario.f_value = 0.0015
    # bsm.run()
    w.configure_traits()


def run_predefined_load_test():
    bsm = BondSlipModel(interaction_type='predefined',
                        material_model='damage-plasticity')
    bsm.eval()
    print bsm.get_sv_hist('s')
    print bsm.get_sv_hist('tau')
    print bsm.get_sv_hist('s_p')
    print bsm.get_sv_hist('z')

if __name__ == '__main__':
    # run_interactive_test_d()
    # run_predefined_load_test()
    run_bond_slip_model_p()
