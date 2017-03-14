'''
Created on Mar 2, 2017

@author: rch
'''
from matplotlib.figure import \
    Figure
from traits.api import \
    HasStrictTraits, Str, \
    Instance,  Event, \
    List,  Range, Int, Float, \
    Property, cached_property, on_trait_change
from traitsui.api import \
    TabularEditor
from traitsui.api import \
    View, Item, UItem, VGroup, VSplit, \
    HSplit
from traitsui.tabular_adapter import TabularAdapter

from util.traits.editors.mpl_figure_editor import \
    MPLFigureEditor
from view.plot2d.viz2d import Viz2D


class Viz2DAdapter(TabularAdapter):
    # List of (Column labels, Column ID).
    columns = [('Label',    'label'),
               ]

#-- Tabular Editor Definition --------------------------------------------

# The tabular editor works in conjunction with an adapter class, derived from
# TabularAdapter.
tabular_editor = TabularEditor(
    adapter=Viz2DAdapter(),
    operations=['delete', 'move', 'edit'],
    # Row titles are not supported in WX:
    drag_move=True,
    auto_update=True,
    selected='selected_viz2d',
)


class VizSheet(HasStrictTraits):
    '''Trait definition.
    '''
    name = Str

    vot_min = Float(0.0)
    vot_max = Float(1.0)
    vot = Range(low='vot_min', high='vot_max', step=0.01,
                enter_set=True, auto_set=False)

    def set_vot(self, vot):
        self.vot_max = max(self.vot_max, vot)
        self.vot = vot

    n_cols = Int(1, label='Number of columns',
                 tooltip='Defines a number of columns within the plot pane',
                 enter_set=True, auto_set=False)

    @on_trait_change('vot,n_cols')
    def replot(self):
        for ax, viz2d in zip(self.axes, self.viz2d_list):
            ax.clear()
            viz2d.plot(ax, self.vot)
        self.data_changed = True

    viz2d_list = List(Viz2D)

    def _viz2d_list_items_changed(self):
        self.replot()

    selected_viz2d = Instance(Viz2D)

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

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        return figure

    # Traits view definition:
    traits_view = View(
        VGroup(
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
                    label='Plot configure',
                    scrollable=True
                ),
            ),
            Item('vot')
        ),
        resizable=True,
        width=0.8, height=0.8,
        buttons=['OK', 'Cancel']
    )

if __name__ == '__main__':

    from view.plot2d.example import mpl1, mpl2, ResponseTracer
    rt = ResponseTracer()

    replot = VizSheet(viz2d_list=[mpl1.viz2d['default'],
                                  mpl2.viz2d['default'],
                                  rt.viz2d['default'],
                                  rt.viz2d['time_profile']])
    replot.configure_traits()
