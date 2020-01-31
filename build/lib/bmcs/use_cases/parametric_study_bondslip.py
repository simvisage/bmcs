'''
Created on 02.01.2017

@author: abaktheer
'''
from bmcs.matmod.bond_slip_model import \
    BondSlipModel, Material, LoadingScenario
from bmcs.matmod.mats_bondslip import \
    MATSEvalFatigue
from bmcs.utils.keyref import \
    KeyRef
from bmcs.view.ui.bmcs_tree_node import \
    BMCSTreeNode
from bmcs.view.window.bmcs_window import \
    BMCSWindow
from traits.api import \
    Instance, Property, \
    List, Str, Trait, Button
from traitsui.api import \
    View, Item, UItem, VGroup, HGroup, spring

import matplotlib.gridspec as gridspec


class UCPStudyElement(BMCSTreeNode):
    '''Class controlling plotting options
    for an instance
    '''
    node_name = Str('<unnamed>')

    color = Trait('black', dict(black='k',
                                cyan='c',
                                green='g',
                                blue='b',
                                yellow='y',
                                magneta='m',
                                red='r')
                  )

    linestyle = Trait('solid', dict(solid='-',
                                    dashed='--',
                                    dash_dot='-.',
                                    dotted=':')
                      )

    tree_view = View(VGroup(Item('node_name', label='label'),
                            Item('linestyle'),
                            Item('color'),
                            label='Plotting options'))

    def plot(self, fig):
        #ax = fig.add_subplot(1, 1, 1)
        self.content.plot(fig, color=self.color_, linestyle=self.linestyle_,
                          label=self.node_name)

    def plot_ax(self, ax1, ax2, ax3):
        self.content.plot_custom(ax1=ax1, ax2=ax2, ax3=ax3,  color=self.color_, linestyle=self.linestyle_,
                                 label=self.node_name)


class UCPStudyElementBMCS(UCPStudyElement):
    node_name = '<unnamed bond_slip>'

    tree_node_list = List(Instance(BMCSTreeNode))

    def _tree_node_list_default(self):
        return [BondSlipModel(mats_eval=MATSEvalFatigue())]

    content = Property(depends_on='tree_node_list')

    def _get_content(self):
        return self.tree_node_list[0]

    def _set_content(self, val):
        self.tree_node_list = [val]


class UCParametricStudy(BMCSTreeNode):
    node_name = Str('Parametric study')

    element_to_add = Trait(
        'BondSlipModel', {'BondSlipModel':   UCPStudyElementBMCS})

    add_element = Button('Add')

    def _add_element_fired(self):
        self.append_node(self.element_to_add_())

    tree_view = View(HGroup(UItem('element_to_add', springy=True),
                            UItem('add_element')),
                     spring
                     )

    tree_node_list = List(Instance(BMCSTreeNode))

    def _tree_node_list_default(self):
        return []

    def plot(self, fig):
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        gs = gridspec.GridSpec(2, 2)
        ax3 = fig.add_subplot(gs[-1, :])

        for node in self.tree_node_list:

            node.plot_ax(ax1, ax2, ax3)

bond_slip_ps = UCParametricStudy()
bond_slip_ps.element_to_add = 'BondSlipModel'
bond_slip_ps.add_element = True
bond_slip_ps.add_element = True

ucc = BMCSTreeNode()
ucc.tree_node_list.append(bond_slip_ps)

mxn_ps_view = BMCSWindow(root=ucc)node
mxn_ps_view.configure_traits()
