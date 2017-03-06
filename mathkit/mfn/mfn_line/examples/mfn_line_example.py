
from math import sin, cos, sqrt

from numpy import linspace, frompyfunc
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, \
    Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
    Dict, Property, cached_property, WeakRef, Delegate, \
    ToolbarButton, on_trait_change, Code, Expression, Button
from traitsui.api import \
    Item, View, HGroup, ListEditor, VGroup, VSplit, Group, HSplit
from traitsui.menu import \
    NoButtons, OKButton, CancelButton, Action, CloseAction, Menu, \
    MenuBar, Separator

from mathkit.mfn import MFnLineArray
from mathkit.mfn.mfn_line.mfn_chaco_editor import MFnChacoEditor
from mathkit.mfn.mfn_line.mfn_matplotlib_editor import MFnMatplotlibEditor
from mathkit.mfn.mfn_line.mfn_plot_adapter import MFnPlotAdapter


a = MFnPlotAdapter(max_size=(300, 200),
                   padding={'top': 0.95,
                            'left': 0.1,
                            'bottom': 0.1,
                            'right': 0.95},
                   line_color=['orange'])


class AnalyticalFunction(HasTraits):

    expression = Expression('x**2', auto_set=False, enter_set=True)
    refresh = Button('redraw')

    def _refresh_fired(self):
        xdata = linspace(0.001, 10, 10000)
        fneval = frompyfunc(lambda x: eval(self.expression), 1, 1)
        ydata = fneval(xdata)
        self.mfn.set(xdata=xdata, ydata=ydata)
        self.mfn.data_changed = True

    mfn = Instance(MFnLineArray)

    def _mfn_default(self):
        return MFnLineArray()

    @on_trait_change('expression')
    def update_mfn(self):
        self._refresh_fired()

    view_mpl = View(HGroup(Item('expression'), Item('refresh')),
                    Item('mfn', editor=MFnMatplotlibEditor(adapter=a),
                         show_label=False),
                    resizable=True,
                    scrollable=True,
                    height=0.5, width=0.5
                    )

    view_chaco = View(HGroup(Item('expression'), Item('refresh')),
                      Item('mfn', editor=MFnChacoEditor(adapter=a),
                           resizable=True, show_label=False),
                      resizable=True,
                      scrollable=True,
                      height=0.3, width=0.3
                      )

if __name__ == '__main__':
    fn = AnalyticalFunction()
    fn._refresh_fired()
    fn.configure_traits(view="view_mpl", kind='nonmodal')
  #  fn.configure_traits( view = "view_chaco", kind = 'nonmodal' )
