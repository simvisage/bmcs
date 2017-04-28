'''
Created on 12.12.2016

@author: abaktheer, rch

@todo: reset loading scenario upon restart
@todo: issue restart upon a change of a material parameter
@todo: more natural representation of polymorphic tree nodes
@todo: loading_scenario - two classes of attributes - categoric and numeric
        check the dependencies between them, so that they can be set.
'''


from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from bmcs.time_functions.tfun_pwl_interactive import TFunPWLInteractive
from ibvpy.api import IMATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import \
    Property, Instance, cached_property, Str, \
    List, Float, Trait, on_trait_change, Bool, Dict,\
    Int, Tuple
from traitsui.api import \
    View, Item, UItem, Group, VGroup, VSplit
from traitsui.editors.enum_editor import EnumEditor
from view.plot2d import Vis2D, Viz2D
from view.ui import BMCSTreeNode
from view.window.bmcs_window import BMCSModel, BMCSWindow
from bmcs.mats.mats_damage_fn import \
    IDamageFn, \
    LiDamageFn, JirasekDamageFn, AbaqusDamageFn
from mats_bondslip import MATSBondSlipD, MATSBondSlipDP, MATSBondSlipEP
import numpy as np


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

    s_0 = Float

    @on_trait_change('tau_bar,E_b')
    def _update_s0(self):
        self.s_0 = self.tau_bar / self.E_b
        self.omega_fn.s_0 = self.s_0

    omega_fn_type = Trait('li',
                          dict(li=LiDamageFn,
                               jirasek=JirasekDamageFn,
                               abaqus=AbaqusDamageFn),
                          MAT=True,
                          )

    @on_trait_change('omega_fn_type,s_0')
    def _reset_omega_fn(self):
        self.omega_fn = self.omega_fn_type_(s_0=self.s_0)

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
            # @todo: fix this to avoid array
            t_ = np.array([t], dtype=np.float_)
            for sv_record, state in zip(sv_records,
                                        (t_, s) + self.state_vars):
                sv_record.append(np.copy(state))

        # append the data to the previous simulation steps
        for sv_name, sv_record in zip(sv_names, sv_records):
            self.sv_hist[sv_name].append(np.array(sv_record))

        self.tline.val = min(t, self.tline.max)

    def get_sv_hist(self, sv_name):
        return np.vstack(self.sv_hist[sv_name])

    viz2d_classes = {'bond history': Viz2DBondHistory,
                     'load function': Viz2DLoadControlFunction,
                     }

    tree_view = View(Item('material_model'),
                     Item('interaction_type'))


def run_bond_slip_model_d(*args, **kw):
    bsm = BondSlipModel(interaction_type='predefined',
                        material_model='damage',
                        )
    w = BMCSWindow(model=bsm, n_cols=2)
    bsm.add_viz2d('bond history', 's-t', x_sv_name='t', y_sv_name='s')
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 'omega-s', x_sv_name='s', y_sv_name='omega')
    bsm.add_viz2d('bond history', 'kappa-s', x_sv_name='s', y_sv_name='kappa')
    bsm.loading_scenario.set(loading_type='cyclic',
                             amplitude_type='constant'
                             )
    bsm.loading_scenario.set(number_of_cycles=1,
                             maximum_loading=0.005,
                             unloading_ratio=0.5)
    bsm.material.omega_fn_type = 'jirasek'
    bsm.material.omega_fn.s_f = 0.003
    bsm.run()
    w.configure_traits(*args, **kw)


def run_bond_slip_model_p(*args, **kw):
    bsm = BondSlipModel(interaction_type='predefined',
                        material_model='plasticity',
                        n_steps=2000)
    w = BMCSWindow(model=bsm, n_cols=2)
    bsm.add_viz2d('bond history', 's-t', x_sv_name='t', y_sv_name='s')
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 's_p-s', x_sv_name='s', y_sv_name='s_p')
    bsm.add_viz2d('bond history', 'z-s', x_sv_name='s', y_sv_name='z')
    bsm.add_viz2d('bond history', 'alpha-s', x_sv_name='s', y_sv_name='alpha')
    bsm.loading_scenario.set(loading_type='cyclic',
                             amplitude_type='constant'
                             )
    bsm.loading_scenario.set(number_of_cycles=1,
                             maximum_loading=0.005)
    bsm.material.set(gamma=0, K=-0)
    bsm.run()
    w.configure_traits(*args, **kw)


def run_bond_slip_model_dp(*args, **kw):
    bsm = BondSlipModel(interaction_type='predefined',
                        material_model='damage-plasticity',
                        n_steps=100,)
    w = BMCSWindow(model=bsm, n_cols=2)
    bsm.add_viz2d('bond history', 's-t', x_sv_name='t', y_sv_name='s')
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 's_p-s', x_sv_name='s', y_sv_name='s_p')
    bsm.add_viz2d('bond history', 'z-s', x_sv_name='s', y_sv_name='z')
    bsm.add_viz2d('bond history', 'alpha-s', x_sv_name='s', y_sv_name='alpha')
    bsm.add_viz2d('bond history', 'omega-s', x_sv_name='s', y_sv_name='omega')
    bsm.loading_scenario.set(loading_type='cyclic',
                             amplitude_type='constant'
                             )
    bsm.loading_scenario.set(maximum_loading=0.005)
    bsm.material.omega_fn_type = 'li'
    bsm.material.set(gamma=0, K=1000)
    bsm.material.omega_fn.set(alpha_1=1.0, alpha_2=2000)
    bsm.run()
    w.configure_traits(*args, **kw)


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
    # run_predefined_load_test(
    run_bond_slip_model_p()
