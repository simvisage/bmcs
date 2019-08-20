'''
Created on 12.08.2019

@author: hspartali
'''
from matplotlib.figure import Figure
from traits.has_traits import HasStrictTraits
from util.traits.editors import MPLFigureEditor

import traits.api as tr
import traitsui.api as ui


class PlotSettings(HasStrictTraits):
    # Plot settings:
    
    figure = tr.Instance(Figure)
    
    columns_headers_list = tr.List([])
    x_axis = tr.Enum(values='columns_headers_list')
    y_axis = tr.Enum(values='columns_headers_list')
    x_axis_multiplier = tr.Enum(1, -1)
    y_axis_multiplier = tr.Enum(-1, 1)
    plot = tr.Button
    
    def _plot_fired(self):
        

    view = ui.View(
        ui.HGroup(ui.Item('x_axis'), ui.Item('x_axis_multiplier')),
        ui.HGroup(ui.Item('y_axis'), ui.Item('y_axis_multiplier')),
        ui.Item('plot', show_label=False)
    )
