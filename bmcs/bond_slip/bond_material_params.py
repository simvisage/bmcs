'''
Created on Apr 28, 2017

@author: abaktheer, rch
'''

from traits.api import \
    Str, \
    Float
from traitsui.api import \
    View, Item, Group, VGroup
from view.ui import BMCSLeafNode


class MaterialParams(BMCSLeafNode):
    '''Record of material parameters of an interface layer
    '''
    node_name = Str('material parameters')

    E_f = Float(240000,
                MAT=True,
                input=True,
                label="E_f ",
                desc="Reinforcement stiffness",
                enter_set=True,
                auto_set=False)

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
