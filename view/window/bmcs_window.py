'''


@author: rch
'''

from traits.api import \
    HasStrictTraits, Instance, Button, Event
from traits.etsconfig.api import ETSConfig
from traitsui.api import \
    TreeEditor, TreeNode, View, Item, VGroup, \
    ToolBar, \
    HSplit
from traitsui.menu import \
    Menu, MenuBar, Separator
from view.ui.bmcs_tree_node import \
    BMCSTreeNode, BMCSLeafNode

from bmcs_plot_dock_pane import PlotDockPane
from bmcs_tree_view_handler import \
    BMCSTreeViewHandler, plot_self, menu_save, \
    menu_open, menu_exit, toolbar_actions


if ETSConfig.toolkit == 'wx':
    from traitsui.wx.tree_editor import \
        DeleteAction
if ETSConfig.toolkit == 'qt4':
    from traitsui.qt4.tree_editor import \
        DeleteAction
else:
    raise ImportError, "tree actions for %s toolkit not availabe" % \
        ETSConfig.toolkit


tree_node = TreeNode(node_for=[BMCSTreeNode],
                     auto_open=False,
                     children='tree_node_list',
                     label='node_name',
                     view='tree_view',
                     menu=Menu(plot_self, DeleteAction),
                     )

leaf_node = TreeNode(node_for=[BMCSLeafNode],
                     auto_open=True,
                     children='',
                     label='node_name',
                     view='tree_view',
                     menu=Menu(plot_self)
                     )

tree_editor = TreeEditor(
    nodes=[tree_node, leaf_node],
    selected='selected_node',
    orientation='vertical'
)


class BMCSWindow(HasStrictTraits):

    '''View object for a cross section state.
    '''
    root = Instance(BMCSTreeNode)

    selected_node = Instance(HasStrictTraits)

    def _selected_node_changed(self):
        self.selected_node.ui = self

    plot_dock_pane = Instance(PlotDockPane, ())

    data_changed = Event

    replot = Button

    def _replot_fired(self):
        self.figure.clear()
        self.selected_node.plot(self.figure)
        self.data_changed = True

    clear = Button()

    def _clear_fired(self):
        self.figure.clear()
        self.data_changed = True

    #time = self.root.time

    view = View(
        HSplit(
            VGroup(
                Item('root',
                     id='bmcs.hsplit.left.tree.id',
                     dock='tab',
                     editor=tree_editor,
                     resizable=True,
                     show_label=False,
                     width=300,
                     height=200,
                     ),
                #                Item('selected_node@'),
                id='bmcs.hsplit.left.id',
                dock='tab',
            ),
            VGroup(
                Item('plot_dock_pane@',
                     show_label=False,
                     id='bmcs.hsplit.viz3d.notebook.id',
                     dock='tab',
                     ),
                # Item('self.root.time', label='t/T_max'),
                dock='tab',
                id='bmcs.hsplit.viz3d.id',
                label='plot sheet',
            ),
            dock='tab',
            id='bmcs.hsplit.id',
        ),
        #        dock='tab',
        id='bmcs.id',
        width=1.0,
        height=1.0,
        title='BMCS',
        resizable=True,
        handler=BMCSTreeViewHandler(),
        toolbar=ToolBar(*toolbar_actions),
        menubar=MenuBar(Menu(menu_exit, Separator(),
                             menu_save, menu_open,
                             name='File'))
    )

if __name__ == '__main__':

    from view.plot2d.example import rt
    from ibvpy.core.bcond_mngr import BCondMngr
    from ibvpy.bcond import BCDof, BCSlice
    bc_mngr = BCondMngr()
    bc_mngr.bcond_list = [BCDof(), BCSlice()]
    tr = BMCSTreeNode(node_name='root',
                      tree_node_list=[BMCSTreeNode(node_name='subnode 1'),
                                      BMCSTreeNode(node_name='subnode 2'),
                                      rt,
                                      bc_mngr
                                      ])

    tv = BMCSWindow(root=tr)
    tv.plot_dock_pane.viz2d_list.append(rt.viz2d['default'])
    tv.configure_traits()
