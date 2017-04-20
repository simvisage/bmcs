'''
Created on 12.12.2016

@author: abaktheer
'''


from bmcs.pullout.pullout import LoadingScenario
from scipy.interpolate import interp1d
from traits.api import \
    HasTraits, Property, Instance, cached_property, Str, Button, Enum, \
    Range, on_trait_change, Array, List, Float
from traitsui.api import \
    View, Item, Group, VGroup, HSplit, TreeEditor, TreeNode
from util.traits.editors import MPLFigureEditor
from view.plot2d import Vis2D, Viz2D
from view.ui import BMCSRootNode, BMCSLeafNode
from view.window.bmcs_window import BMCSModel, BMCSWindow

import matplotlib.gridspec as gridspec
from mats_bondslip_1 import MATSEBondSlipEP
import numpy as np


class Material(BMCSLeafNode):

    node_name = Str('material parameters')
    E_b = Float(12900,
                MAT=True,
                label="G",
                desc="Shear Stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(60,
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
                    label="Tau_pi_bar ",
                    desc="Reversibility limit",
                    enter_set=True,
                    auto_set=False)

    alpha = Float(1.0,
                  MAT=True,
                  )
    beta = Float(1.0,
                 MAT=True,
                 )

    view = View(VGroup(Group(Item('E_b'),
                             Item('tau_bar'), show_border=True, label='Bond Stiffness and reversibility limit'),
                       Group(Item('gamma'),
                             Item('K'), show_border=True, label='Hardening parameters'),
                       Group(Item('alpha'),
                             Item('beta'), label='Damage cumulation parameters')))

    tree_view = view


class Viz2DStressSlip(Viz2D):

    def plot(self, ax, vot, *args, **kw):  # , color='blue', linestyle='-',

        s_arr = self.vis2d.s_arr
        tau_arr = self.vis2d.tau_arr
        tau_arr_e = self.vis2d.tau_arr_e
        ax.plot(s_arr, tau_arr, label='Plastic-Damage')
        ax.plot(s_arr, tau_arr_e, '--k', label='Elastic-Plastic')
        ax.set_title('Slip - Stress')
        ax.set_xlabel('Slip')
        ax.set_ylabel('Stress')
        ax.legend()


class Viz2DDamageSlip(Viz2D):

    def plot(self, ax, vot, *args, **kw):  # , color='blue', linestyle='-',

        s_arr = self.vis2d.s_arr
        omega_arr = self.vis2d.omega_arr

        ax.plot(s_arr, omega_arr)
        ax.set_title('Slip - Damage')
        ax.set_xlabel('Slip')
        ax.set_ylabel('Damage')
        ax.set_ylim([0, 1])


class BondSlipModel(BMCSModel, Vis2D):

    node_name = Str('Bond slip model')

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.material, self.loading_scenario]

    material = Instance(Material)

    def _material_default(self):
        return Material()

    loading_scenario = Instance(LoadingScenario)

    def _loading_scenario_default(self):
        return LoadingScenario()

    mats_eval = Property(Instance(MATSEBondSlipEP), depends_on='MAT')

    @cached_property
    def _get_mats_eval(self):
        return MATSEBondSlipEP(E_b=self.material.E_b,
                               gamma=self.material.gamma,
                               tau_bar=self.material.tau_bar,
                               K=self.material.K,
                               alpha=self.material.alpha,
                               beta=self.material.beta)

    t_arr = Array(np.float_)
    s_arr = Array(np.float_)
    tau_arr = Array(np.float_)
    tau_arr_e = Array(np.float_)
    omega_arr = Array(np.float_)

    def eval(self):
        self.t_arr = self.loading_scenario.xdata
        self.s_arr = self.loading_scenario.ydata
        self.tau_arr, self.tau_arr_e, self.omega_arr = \
            self.mats_eval.get_bond_slip(self.s_arr)
        return

    def paused(self):
        raise NotImplemented

    def stop(self):
        raise NotImplemented

    viz2d_classes = {'bond stress-slip': Viz2DStressSlip,
                     'bond damage-slip': Viz2DDamageSlip}


def run_bond_slip_model():
    bsm = BondSlipModel()
    w = BMCSWindow(model=bsm)
    bsm.add_viz2d('bond stress-slip')
    bsm.add_viz2d('bond damage-slip')
    w.configure_traits()

if __name__ == '__main__':
    run_bond_slip_model()
