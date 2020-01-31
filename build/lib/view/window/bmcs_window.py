'''

@author: rch
'''


from traits.api import \
    HasStrictTraits, Instance, Button, Event, \
    DelegatesTo
from traits.etsconfig.api import ETSConfig
from traitsui.api import \
    TreeEditor, TreeNode, View, Item, VGroup, \
    ToolBar, \
    HSplit
from traitsui.menu import \
    Menu, MenuBar, Separator
from view.ui.bmcs_tree_node import \
    BMCSRootNode, BMCSTreeNode, BMCSLeafNode
from .bmcs_study import BMCSStudy
from .bmcs_tree_view_handler import \
    menu_tools_report_pdf, menu_tools_report_tex, \
    BMCSTreeViewHandler, plot_self, menu_save, \
    menu_open, menu_exit, \
    toolbar_actions, key_bindings


if ETSConfig.toolkit == 'wx':
    from traitsui.wx.tree_editor import \
        DeleteAction
if ETSConfig.toolkit == 'qt4':
    from traitsui.qt4.tree_editor import \
        DeleteAction
else:
    raise ImportError("tree actions for %s toolkit not available" %
                      ETSConfig.toolkit)


tree_node = TreeNode(node_for=[BMCSRootNode, BMCSTreeNode],
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


class BMCSWindow(BMCSStudy):

    selected_node = Instance(HasStrictTraits)

    def _selected_node_changed(self):
        self.selected_node.ui = self

    def get_vot_range(self):
        return self.viz_sheet.get_vot_range()

    vot = DelegatesTo('viz_sheet')

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

    view = View(
        HSplit(
            VGroup(
                Item('sim',
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
                Item('viz_sheet@',
                     show_label=False,
                     id='bmcs.hsplit.viz3d.notebook.id',
                     dock='tab',
                     ),
                dock='tab',
                id='bmcs.hsplit.viz3d.id',
                label='viz sheet',
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
        key_bindings=key_bindings,
        toolbar=ToolBar(*toolbar_actions,
                        image_size=(32, 32),
                        show_tool_names=False,
                        show_divider=True,
                        name='view_toolbar'),
        menubar=MenuBar(Menu(menu_exit, Separator(),
                             menu_save, menu_open,
                             name='File'),
                        Menu(menu_tools_report_tex,
                             menu_tools_report_pdf,
                             name='Tools'),

                        )
    )


if __name__ == '__main__':
    from view.examples.response_tracer import ResponseTracer
    from simulator.api import Simulator
    from ibvpy.core.bcond_mngr import BCondMngr
    from ibvpy.bcond import BCDof, BCSlice
    bc_mngr = BCondMngr()
    bc_mngr.bcond_list = [
        BCDof(),
        BCSlice()]
    rt = ResponseTracer()
    bm = Simulator(node_name='sim',
                   tree_node_list=[BMCSTreeNode(node_name='subnode 1'),
                                   BMCSTreeNode(node_name='subnode 2'),
                                   rt,
                                   bc_mngr
                                   ])
    bm.tree_node_list += [bm.tline]
    tv = BMCSWindow(model=bm)
    rt.add_viz2d('time_profile', 'time profile')
    tv.configure_traits()
