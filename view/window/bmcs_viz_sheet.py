'''
Created on Mar 2, 2017

@author: rch
'''

import os
import tempfile

from matplotlib.figure import \
    Figure
from reporter import ROutputSection
from traits.api import \
    Str, \
    Instance,  Event, Enum, \
    List,  Range, Int, Float, \
    Property, cached_property, \
    on_trait_change, Bool, Button
from traitsui.api import \
    View, Item, UItem, VGroup, VSplit, \
    HSplit, HGroup, TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from util.traits.editors import \
    MPLFigureEditor
from view.plot2d.viz2d import Viz2D

import matplotlib.pyplot as plt
import numpy as np


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


class VizSheet(ROutputSection):
    '''Vieualization sheet
    - controls the time displayed
    - contains several vizualization adapters.
    This class could be called BMCSTV - for watching the time
    dependent response. It can have several channels - in 2D and 3D
    '''

    def __init__(self, *args, **kw):
        super(VizSheet, self).__init__(*args, **kw)
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

    n_cols = Int(2, label='Number of columns',
                 tooltip='Defines a number of columns within the plot pane',
                 enter_set=True, auto_set=False)

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

    def run_finished(self):
        self.skipped_steps = self.monitor_chunk_size
        self.replot()
        self.offline = True

    monitor_chunk_size = Int(10, label='Monitor each # steps')

    skipped_steps = Int(10)

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
        self.data_changed = True
        self.skipped_steps = 0

    viz2d_list = List(Viz2D)

    viz2d_dict = Property

    def _get_viz2d_dict(self):
        return {viz2d.name: viz2d for viz2d in self.viz2d_list}

    def viz2d_list_items_changed(self):
        self.replot()

    def get_subrecords(self):
        return self.viz2d_list

    export_button = Button(label='Export selected diagram')

    def _export_button_fired(self, vot=0):
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ax = fig.add_subplot(111)
        self.selected_viz2d.plot(ax, self.vot)
        fig.show()

    animate_button = Button(label='Animate selected diagram')

    def _animate_button_fired(self):
        tmpdir = tempfile.mkdtemp()
        for i, vot in enumerate(np.linspace(self.animate_from,
                                            self.animate_to,
                                            self.animate_steps)):
            fname = os.path.join(tmpdir, 'step%03d.jpg' % i)
            self.selected_viz2d.savefig_animate(vot, fname,
                                                (self.fig_width,
                                                 self.fig_height))
        print 'animation stored in %s' % tmpdir

    animate_from = Float(0.0, auto_set=False, enter_set=True)
    animate_to = Float(1.0, auto_set=False, enter_set=True)
    animate_steps = Int(30, auto_set=False, enter_set=True)
    fig_width = Float(8.0, auto_set=False, enter_set=True)
    fig_height = Float(5.0, auto_set=False, enter_set=True)

    save_button = Button(label='Save selected diagram')

    selected_viz2d = Instance(Viz2D)

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    axes = Property(List, depends_on='viz2d_list,viz2d_list_items,n_cols')
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

    # Traits view definition:
    traits_view = View(
        VSplit(
            HSplit(
                VGroup(
                    UItem('figure', editor=MPLFigureEditor(),
                          resizable=True,
                          springy=True),
                    scrollable=True,
                    label='Plot panel'
                ),
                VGroup(
                    Item('n_cols', width=100),
                    VSplit(
                        UItem('viz2d_list@',
                              editor=tabular_editor,
                              width=100),
                        UItem('selected_viz2d@',
                              width=100),
                    ),
                    VGroup(
                        HGroup(UItem('export_button',
                                     springy=True, resizable=True)),
                        HGroup(UItem('animate_button',
                                     springy=True, resizable=True)),
                        VGroup(UItem('animate_from', resizable=True),
                               UItem('animate_to', resizable=True),
                               UItem('animate_steps', resizable=True),
                               Item('fig_width'),
                               Item('fig_height'),
                               label='Animation range'
                               ),
                    ),
                    label='Plot configure',
                    scrollable=True
                ),
            ),
            VGroup(
                HGroup(
                    Item('mode'),
                    Item('monitor_chunk_size'),
                ),
                Item('vot_slider', height=40),
            )
        ),
        resizable=True,
        width=0.8, height=0.8,
        buttons=['OK', 'Cancel']
    )


if __name__ == '__main__':

    replot = VizSheet()
    replot.configure_traits()
