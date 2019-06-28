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

# Enthought library imports
import wx
import csv  # , sys

from numpy import *
from scipy.special import jn
from traits.api import \
    Enum, false, Str, Range, Tuple, Bool, Trait, Int, Any, Property, Instance, HasPrivateTraits
from traitsui.api import Item, UI

from etsproxy.enable.api import black_color_trait, LineStyle, ColorTrait, white_color_trait, Window, Component
from etsproxy.kiva.backend_image import GraphicsContext
from etsproxy.kiva.traits.kiva_font_trait import KivaFont
from etsproxy.pyface.api import FileDialog, OK, ImageResource
from etsproxy.pyface.image_resource import ImageResource
from etsproxy.traits.ui.basic_editor_factory import BasicEditorFactory
from etsproxy.traits.ui.menu import Action, ToolBar, Menu
from etsproxy.traits.ui.wx.editor import Editor

from .mfn_plot_adapter import MFnPlotAdapter


USE_DATA_UPDATE = 1
WILDCARD = "Saved plots (*.png)|*.png|"\
           "All files (*.*)|*.*"

# Range for the height and width for the plot widget.
PlotSize = Range(50, 1000, 180)


class _MFnChacoEditor (Editor):
    """ Traits UI editor for displaying trait values in a MFnLine.
    """

    # TableEditorToolbar associated with the editor:
    toolbar = Any
    # The Traits UI associated with the function editor toolbar:
    #  toolbar_ui = Instance( UI )

    # adjustable parameters
    adapter = Instance(MFnPlotAdapter)

    splot = Instance(Plot)
    line_plot = Instance(LinePlot)

    #-------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #-------------------------------------------------------------------------

    plot_container = Instance(OverlayPlotContainer)

    def _plot_container_default(self):
        container = OverlayPlotContainer(padding=50, fill_padding=False,
                                         bgcolor=self.adapter.bgcolor, use_backbuffer=True)

        return container

    #-------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #-------------------------------------------------------------------------
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """

    def init(self, parent):

        factory = self.factory
        self.adapter = factory.adapter

        self.control = self._create_canvas(parent)

        # Register the update listener
        #
        self.value.on_trait_change(self.update_editor, 'data_changed')

    def update_editor(self):
        c = self.plot_container
        c.remove(*c.components)
        self._refresh_container()

    def _create_canvas(self, parent):
        '''Create canvas for chaco plots
        '''
        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)

        container_panel = Window(panel, component=self.plot_container)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticLine(parent, -1, style=wx.LI_HORIZONTAL), 0,
                  wx.EXPAND | wx.BOTTOM, 5)
        sizer.Add(self._create_toolbar(panel), 0, wx.EXPAND)
        sizer.Add(container_panel.control, 1, wx.EXPAND)

        sizer.Add(wx.StaticLine(parent, -1, style=wx.LI_HORIZONTAL), 0,
                  wx.EXPAND | wx.BOTTOM, 5)

        panel.SetSizer(sizer)

        return panel

    def _refresh_container(self):
        ''' rebuild the container for the current data
        '''
        broadcaster = BroadcasterTool()

        mfn_line = self.value
        ydata = transpose(mfn_line.ydata)

        adapter = self.adapter
        if adapter.var_x != '':
            # Get the x-label text from the object's trait var_x
            label_x = getattr(self.object, adapter.var_x)
        else:
            # Get the x-label from the adapter
            label_x = adapter.label_x

        if adapter.var_y != '':
            label_y = getattr(self.object, adapter.var_y)
        else:
            label_y = adapter.label_y

        index = ArrayDataSource(mfn_line.xdata)
        index_range = DataRange1D()
        index_range.add(index)
        index_mapper = LinearMapper(range=index_range)

        value_range = DataRange1D(low_setting=0.0)

        colors = []
        colors = adapter.line_color  # self.line_color_chaco.values()

        styles = []
        styles = adapter.line_style  # self.line_style_chaco.values()

        s_item = list(styles.items())

        color_chaco = []
        style_chaco = []
        c_index = 0  # loop for colors
        s_index = 0  # loop for styles
        i = 0          # for colors and styles
        plots = {}   # for legend

        pd = ArrayPlotData(index=mfn_line.xdata)
        self.splot = Plot(pd)

        for vector in ydata[:]:

            if len(colors) == c_index:
                c_index = 0
            if len(styles) == s_index:
                s_index = 0

            style_name = s_item[s_index][0]
            color_chaco.append(colors[c_index])
            style_chaco.append(style_name)
            c_index = c_index + 1
            s_index = s_index + 1

            y = ArrayDataSource(vector, sort_order="none")

            value_range.add(y)
            value_mapper = LinearMapper(range=value_range)

            self.line_plot = LinePlot(index=index, value=y,
                                      index_mapper=index_mapper,
                                      value_mapper=value_mapper,
                                      color=color_chaco[i],
                                      line_width=adapter.linewidth,
                                      edge_color='blue',
                                      border_visible=False,
                                      line_style=style_chaco[i])

            add_default_grids(self.line_plot)
            add_default_axes(self.line_plot, vtitle=label_y, htitle=label_x)

            self.plot_container.add(self.line_plot)
#            pan = PanTool(line_plot)
#            zoom = SimpleZoom(line_plot, tool_mode="box", always_on=False)
#            broadcaster.tools.append(pan)
#            broadcaster.tools.append(zoom)

            # Add the traits inspector tool to the container
            #
 #           self.plot_container.tools.append(TraitsTool( self.plot_container ))

            self.line_plot.tools.append(PanTool(self.line_plot))
            self.line_plot.overlays.append(ZoomTool(self.line_plot))

            # Legend
            lgnd = adapter.legend_labels[i]
            plots[lgnd] = self.line_plot

            # change the color of the curves
            i = i + 1

        legend = Legend(component=self.plot_container, padding=10, align="ul")
        legend.tools.append(LegendTool(legend, drag_button="right"))
        self.plot_container.overlays.append(legend)

        # Set the list of plots on the legend
        legend.plots = plots

        # Add the title at the top
        self.plot_container.overlays.append(PlotLabel(adapter.title,
                                                      component=self.plot_container,
                                                      font="swiss 16",
                                                      overlay_position="top"))

    #-------------------------------------------------------------------------
    #  Creates the table editing tool bar:
    #-------------------------------------------------------------------------

    def _create_toolbar(self, parent):
        """ Creates the table editing toolbar.
        """
        factory = self.factory

        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        toolbar = MFnChacoEditorToolbar(parent=parent, editor=self)
        tb_sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel.SetSizer(tb_sizer)

        self.toolbar = toolbar
        tb_sizer.Add(toolbar.control, 0)
        tb_sizer.Add((1, 1), 1, wx.EXPAND)

        return panel

    #-------------------------------------------------------------------------
    #  Handles the user requesting that columns not be sorted:
    #-------------------------------------------------------------------------
    def on_savedata(self):
        """ Handles the user requesting that the data of the function is to be saved.
        """
        import os
        dlg = FileDialog(parent=self.control,
                         title='Export function data',
                         default_directory=os.getcwd(),
                         default_filename="", wildcard='*.csv',
                         action='save as')
        if dlg.open() == OK:
            path = dlg.path

            print("Saving data to", path, "...")
            try:
                vectors = []
                x_values = self.value.xdata
                y_values = self.value.ydata
                #savetxt( path, vstack( (x_values, y_values[:,0], y_values[:,1], y_values[:,2]) ).transpose() )

                print('y_values', y_values)
                y_values_tr = y_values.transpose()
                for vector in y_values_tr[:]:
                    vectors.append(vector)

                savetxt(path, vstack((x_values, vectors)).transpose())

            except:
                print("Error saving!")
                raise
            print("Plot saved.")
        return

    def on_savefig(self):
        """ Handles the user requesting that the image of the function is to be saved.
        """
        import os
        dlg = FileDialog(parent=self.control,
                         title='Save as image',
                         default_directory=os.getcwd(),
                         default_filename="", wildcard=WILDCARD,
                         action='save as')
        if dlg.open() == OK:
            path = dlg.path

            print("Saving plot to", path, "...")
            try:
                # Now we create a canvas of the appropriate size and ask it to render
                # our component.  (If we wanted to display this plot in a window, we
                # would not need to create the graphics context ourselves; it would be
                # created for us by the window.)
                size = (650, 400)
                gc = GraphicsContext(size)
                self.plot_container.draw(gc)
                gc.save(path)
            except:
                print("Error saving!")
                raise
            print("Plot saved.")
        return

#-------------------------------------------------------------------------
#  'MFnChacoEditorToolbar' class:
#-------------------------------------------------------------------------


class MFnChacoEditorToolbar (HasPrivateTraits):
    """ Toolbar displayed in table editors.
    """
    #-------------------------------------------------------------------------
    #  Trait definitions:
    #-------------------------------------------------------------------------

    # Do not sort columns:
    save_data = Instance(Action,
                         {'name':    'Save as data',
                          'tooltip': 'Save the function values',
                          'action':  'on_savedata',
                          'enabled': True,
                          'image':   ImageResource('add')})

    # Move current object up one row:
    save_fig = Instance(Action,
                        {'name':    'Save as fig',
                         'tooltip': 'Save as figure',
                         'action':  'on_savefig',
                         'enabled': True,
                         'image':   ImageResource('save')})

    # The table editor that this is the toolbar for:
    editor = Instance(_MFnChacoEditor)

    # The toolbar control:
    control = Any

    #-------------------------------------------------------------------------
    #  Initializes the toolbar for a specified window:
    #-------------------------------------------------------------------------

    def __init__(self, parent=None, **traits):
        super(MFnChacoEditorToolbar, self).__init__(**traits)
        factory = self.editor.factory

        actions = [self.save_data, self.save_fig]
        toolbar = ToolBar(image_size=(16, 16),
                          show_tool_names=False,
                          show_divider=False,
                          *actions)
        self.control = toolbar.create_tool_bar(parent, self)
        self.control.SetBackgroundColour(parent.GetBackgroundColour())

        # fixme: Why do we have to explictly set the size of the toolbar?
        #        Is there some method that needs to be called to do the
        #        layout?
        self.control.SetSize(wx.Size(23 * len(actions), 16))

    #-------------------------------------------------------------------------
    #  PyFace/Traits menu/toolbar controller interface:
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #  Adds a menu item to the menu bar being constructed:
    #-------------------------------------------------------------------------

    def add_to_menu(self, menu_item):
        """ Adds a menu item to the menu bar being constructed.
        """
        pass

    #-------------------------------------------------------------------------
    #  Adds a tool bar item to the tool bar being constructed:
    #-------------------------------------------------------------------------

    def add_to_toolbar(self, toolbar_item):
        """ Adds a toolbar item to the too bar being constructed.
        """
        pass

    #-------------------------------------------------------------------------
    #  Returns whether the menu action should be defined in the user interface:
    #-------------------------------------------------------------------------

    def can_add_to_menu(self, action):
        """ Returns whether the action should be defined in the user interface.
        """
        return True

    #-------------------------------------------------------------------------
    #  Returns whether the toolbar action should be defined in the user
    #  interface:
    #-------------------------------------------------------------------------

    def can_add_to_toolbar(self, action):
        """ Returns whether the toolbar action should be defined in the user
            interface.
        """
        return True

    #-------------------------------------------------------------------------
    #  Performs the action described by a specified Action object:
    #-------------------------------------------------------------------------

    def perform(self, action, action_event=None):
        """ Performs the action described by a specified Action object.
        """
        getattr(self.editor, action.action)()


class MFnChacoEditor(BasicEditorFactory):
    """ Editor factory for plot editors.
    """

    klass = _MFnChacoEditor

    adapter = Instance(MFnPlotAdapter)

    def _adapter_default(self):
        return MFnPlotAdapter()
