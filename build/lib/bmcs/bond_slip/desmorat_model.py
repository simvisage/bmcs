''''
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
from simulator.api import Simulator
from traits.api import \
    Property, Instance, cached_property, Str, \
    List, Float, Trait, on_trait_change, Bool, Dict,\
    Int, Tuple
from traitsui.api import \
    View, Item, Group, VSplit
from traitsui.editors.enum_editor import EnumEditor
from view.plot2d import Vis2D, Viz2D
from view.ui import BMCSTreeNode
from view.window.bmcs_window import BMCSWindow

import numpy as np

from .mats1d_desmorat import MATS1DDesmorat


class Material(BMCSTreeNode):

    node_name = 'material parameters'

    E_b = Float(25000,
                label="E_b",
                MAT=True,
                symbol=r'E_\mathrm{b}',
                unit='MPa/mm',
                desc='elastic bond stiffness',
                enter_set=True,
                auto_set=False)

    E_m = Float(10000,
                label="E_m",
                MAT=True,
                symbol=r'E_\mathrm{m}',
                unit='MPa/mm',
                desc='matrix elastic stiffness',
                enter_set=True,
                auto_set=False)

    gamma = Float(1000,
                  label="gamma",
                  desc="kinematic hardening modulus",
                  MAT=True,
                  symbol=r'\gamma',
                  unit='MPa/mm',
                  enter_set=True,
                  auto_set=False)

    K = Float(3000,
              label="K",
              desc="isotropic hardening modulus",
              MAT=True,
              symbol='K',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    S = Float(0.3,
              label="S",
              desc="damage strength",
              MAT=True,
              symbol='S',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    tau_bar = Float(5,
                    label="tau_bar ",
                    desc="Yield stress",
                    symbol=r'\s{\Tau_0}',
                    unit='MPa',
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    view = View(
        Group(
            VSplit(
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


class DesmoratModel(Simulator, Vis2D):

    node_name = Str('Bond slip model')

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.tline,
                self.mats_eval,
                self.loading_scenario]

    @on_trait_change('MAT,+BC')
    def _update_node_list(self):
        self.tree_node_list = [self.tline,
                               self.mats_eval,
                               self.loading_scenario]

    interaction_type = Trait('predefined',
                             {'interactive': TFunPWLInteractive,
                              'predefined': LoadingScenario},
                             BC=True,
                             symbol='option',
                             unit='-',
                             desc=r'type of loading scenario, possible values:'
                             r'[predefined, interactive]'
                             )
    '''Type of control - either interactive or predefined.
    '''

    def _interaction_type_changed(self):
        self.loading_scenario = self.interaction_type_()

    loading_scenario = Instance(MFnLineArray,
                                report=True,
                                desc=r'object describing the loading scenario as a function')
    '''Loading scenario in form of a function that maps the time variable
    to the control slip.
    '''

    def _loading_scenario_default(self):
        return self.interaction_type_()

    material_model = Trait('plasticity',
                           {'damage-plasticity': MATS1DDesmorat,
                            },
                           enter_set=True, auto_set=False,
                           MAT=True,
                           symbol='option',
                           unit='-',
                           desc=r'type of material model - possible values:'
                           r'[damage-plasticity]'
                           )
    '''Available material models.
    '''

    @on_trait_change('material_model')
    def _set_mats_eval(self):
        self.mats_eval = self.material_model_()
        self._update_node_list()

    mats_eval = Instance(IMATSEval,
                         report=True,
                         desc='object defining the material behavior')
    '''Material model'''

    def _mats_eval_default(self):
        return self.material_model_()

    material = Property

    def _get_material(self):
        return self.mats_eval

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
                  symbol='n_\mathrm{s}',
                  unit='-',
                  desc=r'number of increments',
                  enter_set=True,
                  auto_set=False)

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


def run_bond_slip_model_dp(*args, **kw):
    bsm = DesmoratModel(name='t23_bond_slip_damage_plasticity_based',
                        interaction_type='predefined',
                        material_model='damage-plasticity',
                        n_steps=100,)
    w = BMCSWindow(model=bsm, n_cols=2)
    bsm.add_viz2d('bond history', 's-t', x_sv_name='t', y_sv_name='s')
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 's_p-s', x_sv_name='s', y_sv_name='s_p')
    bsm.add_viz2d('bond history', 'z-s', x_sv_name='s', y_sv_name='z')
    bsm.add_viz2d('bond history', 'alpha-s', x_sv_name='s', y_sv_name='alpha')
    bsm.loading_scenario.trait_set(loading_type='cyclic',
                                   amplitude_type='constant'
                                   )
    bsm.loading_scenario.trait_set(maximum_loading=0.005)
    bsm.material.trait_set(gamma=0, K=1000)
    w.run()
    w.configure_traits(*args, **kw)


def run_interactive_test():
    bsm = DesmoratModel(interaction_type='interactive')
    print('set f_val')
    bsm.loading_scenario.f_value = 0.1
    print('eval')
    bsm.eval()
    print('set f_val')
    bsm.loading_scenario.f_value = 0.4
    print('eval')
    bsm.eval()
    print(list(bsm.sv_hist.keys()))


def run_predefined_load_test():
    bsm = DesmoratModel(interaction_type='predefined',
                        material_model='damage-plasticity')
    bsm.eval()
    print(bsm.get_sv_hist('s'))
    print(bsm.get_sv_hist('tau'))
    print(bsm.get_sv_hist('s_p'))
    print(bsm.get_sv_hist('z'))


if __name__ == '__main__':
    # run_bond_slip_model_dp()
    # run_predefined_load_test(
    # run_bond_slip_model_d()
    from IPython.display import Latex
    import IPython as ip
    bsm = DesmoratModel()
    print((bsm._repr_latex_()))
