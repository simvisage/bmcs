'''
Created on 12.12.2016

@author: abaktheer
'''


from ibvpy.api import IMATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import \
    Property, Instance, cached_property, Str, \
    List, Float, Trait, on_trait_change, Bool, Dict
from traitsui.api import \
    View, Item, Group, VGroup
from traitsui.editors.enum_editor import EnumEditor
from view.examples.tfun_pwl_interactive import TFunPWLInteractive
from view.plot2d import Vis2D, Viz2D
from view.ui import BMCSLeafNode
from view.window.bmcs_window import BMCSModel, BMCSWindow

from bmcs.pullout.pullout import LoadingScenario
from mats_bondslip import MATSBondSlipD, MATSBondSlipDP, MATSBondSlipEP
import numpy as np


class Material(BMCSLeafNode):

    node_name = Str('material parameters')
    E_b = Float(12900,
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

    tau_bar = Float(5,
                    MAT=True,
                    label="Tau_0",
                    desc="yield stress",
                    enter_set=True,
                    auto_set=False)

    alpha = Float(1.0,
                  MAT=True,
                  label="alpha",
                  desc="parameter controls the damage function",
                  enter_set=True,
                  auto_set=False)

    beta = Float(1.0,
                 MAT=True,
                 label="beta",
                 desc="parameter controls the damage function",
                 enter_set=True,
                 auto_set=False)

    view = View(VGroup(Group(Item('E_b'),
                             Item('tau_bar'), show_border=True, label='Bond Stiffness and yield stress'),
                       Group(Item('gamma'),
                             Item('K'), show_border=True, label='Hardening parameters'),
                       Group(Item('alpha'),
                             Item('beta'), label='Damage parameters'),))
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
        ax.set_title('Slip - Stress')
        ax.set_xlabel(self.x_sv_name)
        ax.set_ylabel(self.y_sv_name)
        ax.legend()

    traits_view = View(
        Item('x_sv_name', editor=EnumEditor(name='sv_names')),
        Item('y_sv_name', editor=EnumEditor(name='sv_names'))
    )


class BondSlipModel(BMCSModel, Vis2D):

    node_name = Str('Bond slip model')

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.material,
                self.loading_scenario]

    @on_trait_change('MAT,+BC')
    def _update_node_list(self):
        self.tree_node_list = [self.material, self.loading_scenario]

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
        return self.mats_eval.sv_names

    sv_hist = Dict((Str, List))
    '''Record of state variables. The number of names of the variables
    depend on the active material model. See s_names.
    '''

    def _sv_hist_default(self):
        sv_hist = {}
        for sv_name in ['t', 's'] + self.sv_names:
            sv_hist[sv_name] = []
        return sv_hist

    @on_trait_change('MAT')
    def _sv_hist_reset(self):
        for sv_name in ['t', 's'] + self.sv_names:
            self.sv_hist[sv_name] = []

    def paused(self):
        raise NotImplemented

    def stop(self):
        self._sv_hist_reset()
        raise NotImplemented

    paused = Bool(False)
    restart = Bool(True)

    def init_loop(self):
        if self.paused:
            self.paused = False
        if self.restart:
            self.tline.val = 0
            self.restart = False

    def eval(self):
        '''this method is just called by the tloop_thread'''

        self.init_loop()
        t_min = self.tline.val
        t_max = self.tline.max
        n_steps = 10
        tarray = np.linspace(t_min, t_max, n_steps)
        sv_names = ['t', 's'] + self.sv_names
        sv_records = [[] for s_n in sv_names]

        s_last = 0
        state_vars = self.mats_eval.init_state_vars()
        for idx, t in enumerate(tarray):
            if self.restart or self.paused:
                break
            s = self.loading_scenario(t)
            d_s = s - s_last
            state_vars = \
                self.mats_eval.get_next_state(s, d_s, state_vars)
            self.tline.val = t

            # record the current simulation step
            for sv_record, state in zip(sv_records,
                                        (t, s) + state_vars):
                sv_record.append(np.copy(state))

        # append the data to the previous simulation steps
        for sv_name, sv_record in zip(sv_names, sv_records):
            self.sv_hist[sv_name].append(np.array(sv_record))
        print self.sv_hist['kappa']
        print self.sv_hist['omega']

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
    w.configure_traits()


def run_bond_slip_model_d():
    bsm = BondSlipModel(interaction_type='predefined',
                        material_model='damage')
    w = BMCSWindow(model=bsm, n_cols=1)
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 'omega-s', x_sv_name='s', y_sv_name='omega')
    w.configure_traits()


def run_bond_slip_model_p():
    bsm = BondSlipModel(interaction_type='predefined',
                        material_model='plasticity')
    w = BMCSWindow(model=bsm, n_cols=1)
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 's_p-s', x_sv_name='s', y_sv_name='s_p')
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


def run_predefined_load_test():
    bsm = BondSlipModel(interaction_type='predefined',
                        material_model='damage')
    bsm.eval()
    print bsm.get_sv_hist('s')
    print bsm.get_sv_hist('tau')
    print bsm.get_sv_hist('kappa')
    print bsm.get_sv_hist('omega')

if __name__ == '__main__':
    run_predefined_load_test()
    # run_bond_slip_model_d()
