#-------------------------------------------------------------------------
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
#-------------------------------------------------------------------------

from matplotlib.figure import Figure
from traits.api import Instance, Int, Str
from traitsui.basic_editor_factory import BasicEditorFactory
from util.traits.editors import MPLFigureEditor

from .mfn_plot_adapter import MFnPlotAdapter


class _MFnMatplotlibEditor(MPLFigureEditor):

    # @todo This is not used here, either activate or delete
    # scrollable  = True

    # the plot representation in matplotlib
    figure = Instance(Figure(facecolor=MFnPlotAdapter().padding_bg_color), ())

    # adjustable parameters
    adapter = Instance(MFnPlotAdapter)

    border_size = Int(0)
    # @todo faezeh please make the mapping from the human readable color
    #
    description = Str

    def init(self, parent):

        super(_MFnMatplotlibEditor, self).init(parent)
        factory = self.factory
        self.adapter = factory.adapter

    def update_editor(self):
        figure = self.figure
        axes = figure.add_subplot(111)
        canvas = figure.canvas
        if canvas is None:
            pass
        else:
            figure.delaxes(axes)
            self._refresh_plot()
            figure.canvas.draw()

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # The panel lets us add additional controls.
        a = self.adapter
        fig = self.figure

        panel = super(_MFnMatplotlibEditor, self)._create_canvas(parent)

#         panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
#         sizer = wx.BoxSizer(wx.VERTICAL)
#         panel.SetSizer(sizer)
#
#         # matplotlib commands to create a canvas
#         mpl_control = FigureCanvas(panel, -1, fig)
#         toolbar = NavigationToolbar2Wx(mpl_control)
#
#         sizer.Add(toolbar, 0, wx.EXPAND)
#         sizer.Add(mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW)

        if a.max_size:
            self.figure.canvas.SetMaxSize(a.max_size)
        if a.min_size:
            self.figure.canvas.SetMinSize((a.min_size))

        if a.padding:
            for side, pad in list(a.padding.items()):
                setattr(self.figure.subplotpars, side, pad)

        return panel

    def _refresh_plot(self):

        figure = self.figure

        mfn_line = self.value
        x = mfn_line.xdata
        y = mfn_line.ydata
        axes = figure.add_subplot(111)

        a = self.adapter
        if a.var_x != '':
            label_x = getattr(self.object, a.var_x)
        else:
            label_x = a.label_x

        if a.var_y != '':
            label_y = getattr(self.object, a.var_y)
        else:
            label_y = a.label_y

        styles_m = list(a.line_style.values())
        #  line_color = self.line_color_matplotlib[ a.line_color ]
        line_color = a.line_color[0]
        line_style = styles_m[0]
        # line_style = self.line_style_matplotlib[ a.line_style  ]

        axes.plot(x, y, color=line_color, linewidth=2.,
                  linestyle=line_style)
        axes.set_xlabel(label_x, weight='semibold')
        axes.set_ylabel(label_y, weight='semibold')
        axes.set_title(a.title, size='large', color=a.title_color,
                       weight='bold', position=(.5, 1.03))
        axes.set_axis_bgcolor(color=a.bgcolor)
        axes.ticklabel_format(scilimits=a.scilimits)
        axes.grid(color='gray', linestyle='--', linewidth=0.1, alpha=0.4)
        axes.set_xscale(a.xscale)
        axes.set_yscale(a.yscale)

# class NavigationToolbar2Wx(NavigationToolbar2, wx.ToolBar):
#
#    def _init_toolbar(self):
#        _NTB2_DSAVE    = wx.NewId()
#        self.AddSimpleTool(_NTB2_DSAVE, _load_bitmap('filesave.png'),
#                           'Data Save', 'Save plot data to file')
#        bind(self, wx.EVT_TOOL, self.save, id=_NTB2_DSAVE)


class MFnMatplotlibEditor(BasicEditorFactory):

    klass = _MFnMatplotlibEditor

    adapter = Instance(MFnPlotAdapter)

    def _adapter_default(self):
        return MFnPlotAdapter()
