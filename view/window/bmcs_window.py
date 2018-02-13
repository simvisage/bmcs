'''

@author: rch
'''

from threading import Thread

from ibvpy.core.tline import TLine
from reporter import Reporter
from reporter.reporter import ReportStudy
from traits.api import \
    HasStrictTraits, Instance, Button, Event, \
    DelegatesTo, Bool, Property
from traits.etsconfig.api import ETSConfig
from traitsui.api import \
    TreeEditor, TreeNode, View, Item, VGroup, \
    ToolBar, \
    HSplit
from traitsui.menu import \
    Menu, MenuBar, Separator
from view.ui.bmcs_tree_node import \
    BMCSRootNode, BMCSTreeNode, BMCSLeafNode

from bmcs_model import BMCSModel
from bmcs_tree_view_handler import \
    menu_tools_report_pdf, menu_tools_report_tex, \
    BMCSTreeViewHandler, plot_self, menu_save, \
    menu_open, menu_exit, \
    toolbar_actions, key_bindings
from bmcs_viz_sheet import VizSheet


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

    def __init__(self, study, *args, **kw):
        super(RunThread, self).__init__(*args, **kw)
        self.daemon = True
        self.study = study

    def run(self):
        print 'STARTING THREAD'
        self.study.model.run()
        print 'THREAD ENDED'

    def xrun(self):
        self.study.model.init()
        self.study.start_event = True
        self.study.running = True
        try:
            self.study.model.eval()
        except Exception as e:
            self.study.running = False
            raise
        self.study.running = False
        self.study.finish_event = True

    def pause(self):
        self.study.model.paused = True

    def stop(self):
        self.study.model.restart = True


class BMCSStudy(ReportStudy):
    '''Combine the model with specification of outputs
    '''

    model = Instance(BMCSModel)
    '''Model of the studied phoenomenon.
    '''

    viz_sheet = Instance(VizSheet, ())
    '''Sheet for 2d visualization.
    '''

    input = Property

    def _get_input(self):
        return self.model

    output = Property

    def _get_output(self):
        return self.viz_sheet

    offline = DelegatesTo('viz_sheet')
    n_cols = DelegatesTo('viz_sheet')

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
        '''If the simulation is running disable the run botton,
        enable the pause button and disable changes in all 
        input parameters.
        '''
        self.enable_run = not self.running
        self.enable_pause = self.running
        self.model.set_traits_with_metadata(self.enable_run,
                                            disable_on_run=True)

    start_event = Event
    '''Event announcing the start of the calculation
    '''

    def _start_event_fired(self):
        self.viz_sheet.run_started()

    finish_event = Event
    '''Event announcing the start of the calculation
    '''

    def _finish_event_fired(self):
        self.viz_sheet.run_finished()

    def run(self):
        if self.running:
            return
        self.enable_stop = True
        self.model.run()
        #self.run_thread = RunThread(self)
        print 'run_thread start'
        # self.run_thread.start()
        print 'launched'

    def pause(self):
        self.model.pause()

    def stop(self):
        self.model.stop()
        self.enable_stop = False

    def report_tex(self):
        r = Reporter(report_name=self.model.name,
                     input=self.model,
                     output=self.viz_sheet)
        r.write()
        r.show_tex()

    def report_pdf(self):
        r = Reporter(studies=[self])
        r.write()
        r.show_tex()
        r.run_pdflatex()
        r.show_pdf()

    def add_viz2d(self, clname, name, **kw):
        self.model.add_viz2d(clname, name, **kw)


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
