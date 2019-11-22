from numpy import array, linspace, trapz, arange
from traits.api import Array, Bool, Callable, Enum, Float, Event, HasTraits, \
    Instance, Int, Trait, ToolbarButton, Button, on_trait_change, \
    Property, cached_property, List
from traitsui.api import Item, View, Group, Handler, HGroup

from traitsui.menu import NoButtons, OKButton, CancelButton, Action, CloseAction, Menu, \
    MenuBar, Separator

from .mfn_line import MFnLineArray

import time
import math


class MFnMultiLine(HasTraits):

    # Public Traits
    lines = List(MFnLineArray)

    xdata = Property(List(Array))

    def _get_xdata(self):
        return [mfn.xdata for mfn in self.lines]

    ydata = Property(List(Array))

    def _get_ydata(self):
        return [mfn.ydata for mfn in self.lines]

    data_changed = Event
