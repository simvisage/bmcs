#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Created on May 25, 2009 by Rostislav Chudoba
#
#-------------------------------------------------------------------------------

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import wx
from numpy import *

from traitsui.api import View, Item
from traits.api import Any, Instance, on_trait_change
from etsproxy.traits.ui.wx.editor import Editor
from etsproxy.traits.ui.basic_editor_factory import BasicEditorFactory
from .mfn_plot_adapter import MFnMultiPlotAdapter
from matplotlib.ticker import MaxNLocator

class _MFnMatplotlibEditor(Editor):
    
    
    # @todo This is not used here, either activate or delete
    scrollable  = True
    
    # adjustable parameters
    adapter = Instance( MFnMultiPlotAdapter )
    
    # the plot representation in matplotlib
    figure = Instance(Figure(facecolor = MFnMultiPlotAdapter().bgcolor), ())

    def init(self, parent):
        
        factory = self.factory
        self.adapter = factory.adapter
        
        self.control = self._create_canvas(parent)
        self.value.on_trait_change( self.update_editor, 'data_changed' )

    def update_editor(self):
        figure = self.figure
        axes = figure.add_subplot(111)
        canvas = figure.canvas
        if canvas is None:
            pass
        else:
            figure.delaxes( axes )
            self._refresh_plot()
            figure.canvas.draw()

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # The panel lets us add additional controls.
        fig = self.figure
    
        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
#
        # matplotlib commands to create a canvas
        mpl_control = FigureCanvas( panel, -1, fig )
        toolbar = NavigationToolbar2Wx(mpl_control)
        sizer.Add(toolbar, 0, wx.EXPAND)
        sizer.Add(mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW )
        fig.canvas.SetMinSize((100,100))
        return panel

    def _refresh_plot( self ):

        a = self.adapter
        figure = self.figure
        mfn_multiline = self.value
        plot_set = []
        for data in mfn_multiline.xdata:
            plot_set.append(any(array(data != [0.,1.])))
        idx = where(array(plot_set) == True)[0]
        xdata = []
        ydata = []
        for index in idx:
            xdata.append(mfn_multiline.xdata[index]),
            ydata.append(mfn_multiline.ydata[index])
        legend = array(a.legend_labels)[idx]
        color = array(a.mline_color)[idx]
        style = array(a.mline_style)[idx]
        width = array(a.mline_width)[idx]
        
        if a.var_x != '':
            label_x = getattr( self.object, a.var_x )
        else:
            label_x = a.label_x
            
        if a.var_y  != '':
            label_y = getattr( self.object, a.var_y )
        else:
            label_y = a.label_y
            
        axes = figure.add_subplot(111)
        
        for x,y,c,s,w in zip(xdata[:], ydata[:], \
                         color[:], style[:], width[:]):
            axes.plot(x, y, color = c, linestyle = s, linewidth = w)
        axes.set_xlabel( a.label_x, weight = 'semibold')
        axes.set_ylabel( a.label_y, weight = 'semibold')
        axes.set_title( a.title,\
                        size = 'large', color = 'black',\
                        weight = 'bold', position = (.5,1.03))
        axes.set_axis_bgcolor(color = 'white')
        axes.ticklabel_format(scilimits = a.scilimits)
        axes.grid(color='gray', linestyle='--', linewidth=0.1, alpha = 0.4)
        axes.legend((legend), loc = 'best' )
        axes.set_xscale(a.xscale)
        axes.set_yscale(a.yscale)
        if a.xticks == 0:
            pass
        else:
            axes.xaxis.set_major_locator(MaxNLocator(a.xticks))
        if a.yticks == 0:
            pass
        else:
            axes.yaxis.set_major_locator(MaxNLocator(a.yticks))


class MFnMatplotlibEditor(BasicEditorFactory):

    klass = _MFnMatplotlibEditor
    
    adapter = Instance( MFnMultiPlotAdapter )
    def _adapter_default(self):
        return MFnMultiPlotAdapter()
    