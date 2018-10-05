
from traits.api import Float, Callable, on_trait_change
from traitsui.api import View, Item
from view.ui import BMCSLeafNode


class TLine(BMCSLeafNode):

    '''
    Time line for the control parameter.

    This class sets the time-range of the computation - the start and stop time.
    val is the value of the current time.

    TODO - the info page including the number of load steps
    and estimated computation time.

    TODO - the slide bar is not read-only. How to include a real progress bar?
    '''
    node_name = 'time range'

    min = Float(0.0,
                TIME=True
                )
    max = Float(1.0,
                TIME=True
                )
    step = Float(0.1,
                 TIME=True
                 )
    val = Float(0.0)

    def _val_changed(self):
        if self.time_change_notifier:
            self.time_change_notifier(self.val)

    @on_trait_change('min,max')
    def _time_range_changed(self):
        if self.time_range_change_notifier:
            self.time_range_change_notifier(self.max)

    time_change_notifier = Callable
    time_range_change_notifier = Callable

    tree_view = View(
        Item('min', full_size=True),
        Item('max'),
        Item('step'),
        Item('val', style='readonly')
    )
