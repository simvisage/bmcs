'''
Created on Mar 2, 2017

@author: rch
'''

from __builtin__ import len
import os
import tempfile
from threading import Thread

from matplotlib.figure import \
    Figure
from mayavi.core.ui.api import \
    MayaviScene, SceneEditor, MlabSceneModel
from pyface.api import GUI
from reporter import ROutputSection
from traits.api import \
    Str, Instance,  Event, Enum, \
    List, Range, Int, Float, \
    Property, cached_property, \
    on_trait_change, Bool, Button, Directory
from traitsui.api import \
    View, Item, UItem, VGroup, VSplit, \
    HSplit, HGroup, Tabbed, TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from util.traits.editors import \
    MPLFigureEditor
from view.plot2d.viz2d import Viz2D

import matplotlib.pyplot as plt
import numpy as np
import traits.api as tr
from view.plot3d.viz3d import Viz3D


class RunThread(Thread):
    '''Time loop thread responsible.
    '''

    def __init__(self, vs, vot, *args, **kw):
        super(RunThread, self).__init__(*args, **kw)
        self.daemon = True
        self.vs = vs
        self.vot = vot

    def run(self):
        print 'STARTING THREAD'
        GUI.invoke_later(self.vs.update_pipeline, self.vot)
        print 'THREAD ENDED'


class Viz2DAdapter(TabularAdapter):
    # List of (Column labels, Column ID).
    columns = [('Label',    'label'),
               ]

#-- Tabular Editor Definition --------------------------------------------


# The tabular editor works in conjunction with an adapter class, derived from
tabular_editor = TabularEditor(
    adapter=Viz2DAdapter(),
    operations=['delete', 'move', 'edit'],
    # Row titles are not supported in WX:
    drag_move=True,
    auto_update=True,
    selected='selected_viz2d',
)


class BMCSVizSheet(ROutputSection):
    '''Vieualization sheet
    - controls the time displayed
    - contains several vizualization adapters.
    This class could be called BMCSTV - for watching the time
    dependent response. It can have several channels - in 2D and 3D
    '''

    def __init__(self, *args, **kw):
        super(BMCSVizSheet, self).__init__(*args, **kw)
        self.on_trait_change(self.viz2d_list_items_changed,
                             'viz2d_list_items')

    name = Str

    min = Float(0.0)
    '''Simulation start is always 0.0
    '''
    max = Float(1.0)
    '''Upper range limit of the current simulator.
    This range is determined by the the time-loop range
    of the model. 
    '''
    vot = Float

    def _vot_default(self):
        return self.min

    vot_slider = Range(low='min', high='max', step=0.01,
                       enter_set=True, auto_set=False)
    '''Time line controlling the current state of the simulation.
    this value is synchronized with the control time of the
    time loop setting the tline. The vot_max = tline.max.
    The value of vot follows the value of tline.val in monitoring mode.
    By default, the monitoring mode is active with vot = tline.value.
    When sliding to a value vot < tline.value, the browser mode is activated.
    When sliding into the range vot > tline.value the monitoring mode
    is reactivated. 
    '''

    def _vot_slider_default(self):
        return 0.0

    mode = Enum('monitor', 'browse')

    def _mode_changed(self):
        if self.mode == 'browse':
            self.offline = False

    time = Float(0.0)

    def time_range_changed(self, max_):
        self.max = max_

    def time_changed(self, time):
        self.time = time
        if self.mode == 'monitor':
            self.vot = time
            self.vot_slider = time

    def _vot_slider_changed(self):
        if self.mode == 'browse':
            if self.vot_slider >= self.time:
                self.mode = 'monitor'
                self.vot_slider = self.time
                self.vot = self.time
            else:
                self.vot = self.vot_slider
        elif self.mode == 'monitor':
            if self.vot_slider < self.time:
                self.mode = 'browse'
                self.vot = self.vot_slider
            else:
                self.vot_slider = self.time
                self.vot = self.time

    offline = Bool(True)
    '''If the sheet is offline, the plot refresh is inactive.
    The sheet starts in offline mode and is activated once the signal
    run_started has been received. Upon run_finished the 
    the sheet goes directly into the offline mode again.
    
    If the user switches to browser mode, the vizsheet gets online 
    and reploting is activated.
    '''

    def run_started(self):
        self.offline = False
        for ax, viz2d in zip(self.axes, self.viz2d_list):
            ax.clear()
            viz2d.reset(ax)
        self.mode = 'monitor'
        if self.reference_viz2d:
            ax = self.reference_axes
            ax.clear()
            self.reference_viz2d.reset(ax)

    def run_finished(self):
        self.skipped_steps = self.monitor_chunk_size
        self.replot()
        # self.update_pipeline(1.0)
        self.offline = True

    monitor_chunk_size = Int(10, label='Monitor each # steps')

    skipped_steps = Int(1)

    @on_trait_change('vot,n_cols')
    def replot(self):
        if self.offline:
            return
        if self.mode == 'monitor' and \
                self.skipped_steps < (self.monitor_chunk_size - 1):
            self.skipped_steps += 1
            return
        for ax, viz2d in zip(self.axes, self.viz2d_list):
            ax.clear()
            viz2d.plot(ax, self.vot)
        if self.reference_viz2d:
            ax = self.reference_axes
            ax.clear()
            self.reference_viz2d.plot(ax, self.vot)
        self.data_changed = True
        self.skipped_steps = 0
        if self.mode == 'browse':
            self.update_pipeline(self.vot)
        else:
            up = RunThread(self, self.vot)
            up.start()

    viz2d_list = List(Viz2D)
    '''List of visualization adaptors for 2D.
    '''
    viz2d_dict = Property

    def _get_viz2d_dict(self):
        return {viz2d.name: viz2d for viz2d in self.viz2d_list}

    viz2d_names = Property
    '''Names to be supplied to the selector of the
    reference graph.
    '''

    def _get_viz2d_names(self):
        return self.viz2d_dict.keys()

    def viz2d_list_items_changed(self):
        self.replot()

    def get_subrecords(self):
        '''What is this good for?
        '''
        return self.viz2d_list

    export_button = Button(label='Export selected diagram')

    def _export_button_fired(self, vot=0):
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ax = fig.add_subplot(111)
        self.selected_viz2d.plot(ax, self.vot)
        fig.show()

    export_path = Directory
    status_message = Str('')

    animate_button = Button(label='Animate selected diagram')

    def _animate_button_fired(self):

        if self.export_path == '':
            dir_ = tempfile.mkdtemp()
        else:
            dir_ = self.export_path
        name = self.selected_viz2d.name
        path = os.path.join(dir_, name)

        if os.path.exists(path):
            self.status_message = 'overwriting animation %s' % name
        else:
            os.makedirs(path)

        for i, vot in enumerate(np.linspace(self.animate_from,
                                            self.animate_to,
                                            self.animate_steps)):
            fname = os.path.join(path, 'step%03d.jpg' % i)
            self.selected_viz2d.savefig_animate(vot, fname,
                                                (self.fig_width,
                                                 self.fig_height))
        self.status_message = 'animation stored in %s' % path

    animate_from = Float(0.0, auto_set=False, enter_set=True)
    animate_to = Float(1.0, auto_set=False, enter_set=True)
    animate_steps = Int(30, auto_set=False, enter_set=True)
    fig_width = Float(8.0, auto_set=False, enter_set=True)
    fig_height = Float(5.0, auto_set=False, enter_set=True)

    save_button = Button(label='Save selected diagram')

    #=========================================================================
    # Reference figure serving for orientation.
    #=========================================================================
    reference_viz2d_name = Enum('', values="viz2d_names")
    '''Current name of the reference graphs.
    '''
    reference_viz2d = Property(Instance(Viz2D),
                               depends_on='reference_viz2d_name')
    '''Visualization of a graph showing the time context of the
    current visualization state. 
    '''

    def _get_reference_viz2d(self):
        if self.reference_viz2d_name == None:
            if len(self.viz2d_dict):
                return self.viz2d_list[0]
            else:
                return None
        return self.viz2d_dict[self.reference_viz2d_name]

    reference_figure = Instance(Figure)

    def _reference_figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    reference_axes = Property(List,
                              depends_on='reference_viz2d_name')
    '''Derived axes objects reflecting the layout of plot pane
    and the individual. 
    '''
    @cached_property
    def _get_reference_axes(self):
        return self.reference_figure.add_subplot(1, 1, 1)

    #=========================================================================
    # Parameters of the current viz2d
    #=========================================================================
    selected_viz2d = Instance(Viz2D)

    n_cols = Int(2, label='Number of columns',
                 tooltip='Defines a number of columns within the plot pane',
                 enter_set=True, auto_set=False)

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    axes = Property(List,
                    depends_on='viz2d_list,viz2d_list_items,n_cols')
    '''Derived axes objects reflecting the layout of plot pane
    and the individual. 
    '''
    @cached_property
    def _get_axes(self):
        n_fig = len(self.viz2d_list)
        n_cols = self.n_cols
        n_rows = (n_fig + n_cols - 1) / self.n_cols
        self.figure.clear()
        return [self.figure.add_subplot(n_rows, self.n_cols, i + 1)
                for i in range(n_fig)]

    data_changed = Event

    bgcolor = tr.Tuple(1.0, 1.0, 1.0)
    fgcolor = tr.Tuple(0.0, 0.0, 0.0)

    scene = Instance(MlabSceneModel)

    def _scene_default(self):
        return MlabSceneModel()

    mlab = Property(depends_on='input_change')
    '''Get the mlab handle'''

    def _get_mlab(self):
        return self.scene.mlab

    fig = Property()
    '''Figure for 3D visualization.
    '''
    @cached_property
    def _get_fig(self):
        fig = self.mlab.gcf()
        bgcolor = tuple(self.bgcolor)
        fgcolor = tuple(self.fgcolor)
        self.mlab.figure(fig, fgcolor=fgcolor, bgcolor=bgcolor)
        return fig

    def show(self, *args, **kw):
        '''Render the visualization.
        '''
        self.mlab.show(*args, **kw)

    def add_viz3d(self, viz3d, order=1):
        '''Add a new visualization objectk.'''
        viz3d.ftv = self
        vis3d = viz3d.vis3d
        label = '%s[%s:%s]-%s' % (viz3d.label,
                                  str(vis3d.__class__),
                                  str(viz3d.__class__),
                                  vis3d
                                  )
        if self.viz3d_dict.has_key(label):
            raise KeyError, 'viz3d object named %s already registered' % label
        viz3d.order = order
        self.viz3d_dict[label] = viz3d

    viz3d_dict = tr.Dict(tr.Str, tr.Instance(Viz3D))
    '''Dictionary of visualization objects.
    '''

    viz3d_list = tr.Property

    def _get_viz3d_list(self):
        map_order_viz3d = {}
        for idx, (viz3d) in enumerate(self.viz3d_dict.values()):
            order = viz3d.order
            map_order_viz3d['%5g%5g' % (order, idx)] = viz3d
        return [map_order_viz3d[key] for key in sorted(map_order_viz3d.keys())]

    pipeline_ready = Bool(False)

    def setup_pipeline(self):
        if self.pipeline_ready:
            return
        self.fig
        fig = self.mlab.gcf()
        fig.scene.disable_render = True
        for viz3d in self.viz3d_list:
            viz3d.setup()
        fig.scene.disable_render = False
        self.pipeline_ready = True

    def update_pipeline(self, vot):
        self.setup_pipeline()
        # get the current constrain information
        self.vot = vot
        fig = self.mlab.gcf()
        fig.scene.disable_render = True
        for viz3d in self.viz3d_list:
            viz3d.plot(vot)
        fig.scene.disable_render = False

    # Traits view definition:
    traits_view = View(
        VSplit(
            HSplit(
                Tabbed(
                    UItem('scene',
                          editor=SceneEditor(scene_class=MayaviScene)),
                    UItem('figure', editor=MPLFigureEditor(),
                          resizable=True,
                          springy=True),
                    scrollable=True,
                    label='Plot panel'
                ),
                VGroup(
                    Item('n_cols', width=250),
                    VSplit(
                        UItem('viz2d_list@',
                              editor=tabular_editor,
                              width=100),
                        UItem('selected_viz2d@',
                              width=200),
                        VGroup(
                            UItem('export_button',
                                  springy=False, resizable=True),
                            VGroup(
                                HGroup(
                                    UItem('fig_width', springy=True,
                                          resizable=False),
                                    UItem('fig_height', springy=True),
                                ),
                                label='Figure size'
                            ),
                            UItem('animate_button',
                                  springy=False, resizable=True),
                            VGroup(
                                HGroup(
                                    UItem('animate_from', springy=True),
                                    UItem('animate_to', springy=True),
                                    UItem('animate_steps', springy=True),
                                ),
                                label='Animation range'
                            ),
                            Item('export_path'),
                            HGroup(
                                UItem('status_message', style='readonly')
                            ),
                        ),
                        VGroup(
                            UItem('reference_viz2d_name', resizable=True),
                            UItem('reference_figure', editor=MPLFigureEditor(),
                                  width=200,
                                  # springy=True
                                  ),
                            label='Reference graph',
                        )
                    ),
                    label='Plot configure',
                    scrollable=True
                ),
            ),
            VGroup(
                HGroup(
                    Item('mode', resizable=False, springy=False),
                    Item('monitor_chunk_size', resizable=False, springy=False),
                ),
                Item('vot_slider', height=40),
            )
        ),
        resizable=True,
        width=0.8, height=0.8,
        buttons=['OK', 'Cancel']
    )


if __name__ == '__main__':
    import traits.api as tr
    from ibvpy.dots.vdots_grid import DOTSGrid
    from ibvpy.fets.fets2D.vfets2D4q import FETS2D4Q
    dots = DOTSGrid(n_x=5, n_y=2, fets=FETS2D4Q())

    n_elems = dots.mesh.n_active_elems
    U = np.random.random(n_elems * 4 * 3).reshape(-1, 3) * 10.0

    class TLoop(tr.HasTraits):

        U_record = tr.List()
        ts = tr.Instance(dots)

    tl = TLoop(ts=dots)
    tl.U_record.append(U)
    viz3d = Viz3D()
    viz3d.set_tloop(tloop=tl)

    vs = BMCSVizSheet()
    vs.add_viz3d(viz3d)
    vs.run_started()
    vs.replot()
    vs.run_finished()
    vs.configure_traits()
