'''
Created on 02.01.2017

@author: abaktheer
'''
from bmcs.matmod.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from bmcs.matmod.mats_bondslip import MATSEvalFatigue
from bmcs.matmod.pull_out_simulation import \
    PullOutSimulation, Material, LoadingScenario, Geometry
from bmcs.matmod.tloop import TLoop
from bmcs.matmod.tstepper import TStepper
from bmcs.utils.keyref import \
    KeyRef
from bmcs.view.ui.bmcs_tree_node import \
    BMCSTreeNode
from bmcs.view.window.bmcs_window import \
    BMCSWindow
from ibvpy.api import BCDof
from traits.api import \
    Instance, Property, \
    List, Str, Trait, Button, cached_property
from traitsui.api import \
    View, Item, UItem, VGroup, HGroup, spring

import matplotlib.gridspec as gridspec


#from bmcs.matmod.tloop import TLoop
#from bmcs.matmod.tstepper import TStepper
#from matmod.pull_out_explorer import loading_scenario
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
        self.content.plot_custom(ax1=ax1, ax2=ax2, ax3=ax3, color=self.color_, linestyle=self.linestyle_,
                                 label=self.node_name)


class UCPStudyElementBMCS_LoadControl(UCPStudyElement):
    node_name = '<unnamed pull_out_load_control>'

    tree_node_list = List(Instance(BMCSTreeNode))

    def _tree_node_list_default(self):

        loading_scenario = LoadingScenario()
        bc_list = [BCDof(var='u', dof=0, value=0.0), BCDof(
            var='f', dof=-1, time_function=loading_scenario.time_func)]
        geometry = Geometry()
        model = PullOutSimulation(geometry=geometry,
                                  loading_scenario=loading_scenario)
        model.time_stepper.bc_list = bc_list
        return [model]

    content = Property(depends_on='tree_node_list')

    def _get_content(self):
        return self.tree_node_list[0]

    def _set_content(self, val):
        self.tree_node_list = [val]


class UCPStudyElementBMCS_DisplacementControl(UCPStudyElement):
    node_name = '<unnamed pull_out_displacement_control>'

    tree_node_list = List(Instance(BMCSTreeNode))

    def _tree_node_list_default(self):

        loading_scenario = LoadingScenario()
        bc_list = [BCDof(var='u', dof=0, value=0.0), BCDof(
            var='u', dof=-1, time_function=loading_scenario.time_func)]
        geometry = Geometry()
        model = PullOutSimulation(geometry=geometry,
                                  loading_scenario=loading_scenario)
        model.time_stepper.bc_list = bc_list
        return [model]

    content = Property(depends_on='tree_node_list')

    def _get_content(self):
        return self.tree_node_list[0]

    def _set_content(self, val):
        self.tree_node_list = [val]


class UCParametricStudy(BMCSTreeNode):
    node_name = Str('Parametric study')

    element_to_add = Trait(
        'PullOutSimulation_Load_Control', {'PullOutSimulation_Load_Control':   UCPStudyElementBMCS_LoadControl,
                                           'PullOutSimulation_Displacement_Control':   UCPStudyElementBMCS_DisplacementControl})

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


pull_out_ps = UCParametricStudy()
pull_out_ps.element_to_add = 'PullOutSimulation_Load_Control'
pull_out_ps.add_element = True
pull_out_ps.element_to_add = 'PullOutSimulation_Displacement_Control'
pull_out_ps.add_element = True


ucc = BMCSTreeNode()
ucc.tree_node_list.append(pull_out_ps)

mxn_ps_view = BMCSWindow(root=ucc)
mxn_ps_view.configure_traits()
