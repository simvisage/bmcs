'''

@author: rch
'''

from threading import Thread

from traits.api import \
    HasStrictTraits, Instance, Button, Event, \
    DelegatesTo, Bool
from traits.etsconfig.api import ETSConfig
from traitsui.api import \
    TreeEditor, TreeNode, View, Item, VGroup, \
    ToolBar, \
    HSplit
from traitsui.menu import \
    Menu, MenuBar, Separator
from view.plot2d import Vis2D
from view.ui.bmcs_tree_node import \
    BMCSRootNode, BMCSTreeNode, BMCSLeafNode

from bmcs_tree_view_handler import \
    BMCSTreeViewHandler, plot_self, menu_save, \
    menu_open, menu_exit, toolbar_actions, key_bindings
from bmcs_viz_sheet import VizSheet
from tline import TLine


if ETSConfig.toolkit == 'wx':
    from traitsui.wx.tree_editor import \
        DeleteAction
if ETSConfig.toolkit == 'qt4':
    from traitsui.qt4.tree_editor import \
        DeleteAction
else:
    raise ImportError, "tree actions for %s toolkit not available" % \
        ETSConfig.toolkit


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


class RunThread(Thread):
    '''Time loop thread responsible.
    '''

    def __init__(self, ui, *args, **kw):
        super(RunThread, self).__init__(*args, **kw)
        self.daemon = True
        self.ui = ui

    def run(self):
        self.ui.start_event = True
        self.ui.running = True
        try:
            self.ui.model.eval()
        except Exception as e:
            self.ui.running = False
            raise
        self.ui.running = False
        self.ui.finish_event = True

    def pause(self):
        self.ui.model.paused = True

    def stop(self):
        self.ui.model.restart = True


class BMCSModel(BMCSRootNode):

    tline = Instance(TLine)

    def _tline_default(self):
        return TLine()

    def eval(self):
        return

    def paused(self):
        pass

    def stop(self):
        pass


class BMCSWindow(HasStrictTraits):

    '''View object for a cross section state.
    '''
    model = Instance(BMCSModel)

    offline = DelegatesTo('viz_sheet')

    def _model_changed(self):
        self.model.set_ui_recursively(self)
        tline = self.model.tline
        self.viz_sheet.time_range_changed(tline.max)
        self.viz_sheet.time_changed(tline.val)

    run_thread = Instance(RunThread)

    running = Bool(False)

    enable_run = Bool(True)
    enable_pause = Bool(False)
    enable_stop = Bool(False)

    def _running_changed(self):
        self.enable_run = not self.running
        self.enable_pause = self.running
        self.enable_stop = self.running
        self.model.set_traits_with_metadata(self.enable_run,
                                            disable_on_run=True)

    start_event = Event

    def _start_event_fired(self):
        self.viz_sheet.run_started()

    finish_event = Event

    def _finish_event_fired(self):
        self.viz_sheet.run_finished()

    def run(self):
        if self.running:
            return
        self.run_thread = RunThread(ui=self)
        print 'STARTING THREAD'
        self.run_thread.start()

    def pause(self):
        self.model.pause()

    def stop(self):
        self.model.stop()

    selected_node = Instance(HasStrictTraits)

    def _selected_node_changed(self):
        self.selected_node.ui = self

    def get_vot_range(self):
        return self.viz_sheet.get_vot_range()

    def set_vot(self, vot):
        self.viz_sheet.set_vot(vot)

    vot = DelegatesTo('viz_sheet')

    viz_sheet = Instance(VizSheet, ())

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
                Item('model',
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
        height=0.9,
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
                             name='File'))
    )

if __name__ == '__main__':

    from view.examples.response_tracer import ResponseTracer
    from ibvpy.core.bcond_mngr import BCondMngr
    from ibvpy.bcond import BCDof, BCSlice
    bc_mngr = BCondMngr()
    bc_mngr.bcond_list = [
        BCDof(),
        BCSlice()]
    rt = ResponseTracer()
    tr = BMCSModel(node_name='model',
                   tree_node_list=[BMCSTreeNode(node_name='subnode 1'),
                                   BMCSTreeNode(node_name='subnode 2'),
                                   rt,
                                   bc_mngr
                                   ])
    tr.tree_node_list += [tr.tline]
    tv = BMCSWindow(model=tr)
    rt.add_viz2d('time_profile')
    tv.configure_traits()
