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
import csv
import os
import sys
import wx

from chaco.api import \
    add_default_axes, \
    add_default_grids, OverlayPlotContainer, \
    PlotGraphicsContext, \
    ArrayDataSource, \
    DataRange1D, LinePlot, LinearMapper
from chaco.tools.api import \
    PanTool, \
    BroadcasterTool, ZoomTool
from enable.api import \
    Window
from pyface.api import FileDialog, OK
from traits.api import \
    Range,  \
    Any,  Instance, HasPrivateTraits
from traitsui.basic_editor_factory import \
    BasicEditorFactory
from traitsui.menu import \
    Action, ToolBar
from traitsui.wx.editor import \
    Editor

from etsproxy.pyface.api \
    import ImageResource
from .mfn_plot_adapter import \
    MFnPlotAdapter
import numpy as np


# from kiva.traits.kiva_font_trait import \
#     KivaFont
#from util.resource import find_resource
# from etsproxy.traits.ui.editor_factory import \
#    EditorFactory
USE_DATA_UPDATE = 1
WILDCARD = "Saved plots (*.png)|*.png|(*.jpg)|*.jpg|"\
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

    lplot = Instance(LinePlot)

    #---------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------

    plot_container = Instance(OverlayPlotContainer)

    def _plot_container_default(self):
        container = OverlayPlotContainer(padding=50,
                                         fill_padding=False,
                                         bgcolor="white",
                                         use_backbuffer=True)

        return container

    #---------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------
        """ Finishes initializing the editor by creating the 
            underlying toolkit widget.
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
#        c.padding = 30
        c.remove(*c.components)
        self._refresh_container()

    def _create_canvas(self, parent):
        '''Create canvas for chaco plots
        '''
        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
#        self.panel = panel

        container_panel = Window(panel, component=self.plot_container)

        a = self.adapter
        if a.max_size:
            panel.SetMaxSize(a.max_size)
        if a.min_size:
            panel.SetMinSize(a.min_size)

        print('size', panel.GetSize())

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticLine(parent, -1, style=wx.LI_HORIZONTAL), 0,
                  wx.EXPAND | wx.BOTTOM, 5)
        sizer.Add(self._create_toolbar(panel), 0, wx.EXPAND)
        sizer.Add(container_panel.control, 1, wx.EXPAND)

        sizer.Add(wx.StaticLine(parent, -1, style=wx.LI_HORIZONTAL), 0,
                  wx.EXPAND | wx.BOTTOM, 5)

        panel.SetSizer(sizer)

        if a.padding:
            for side, pad in list(a.padding.items()):
                if side == 'right' or side == 'top':
                    padd = int((1 - pad) * 200)
                    setattr(self.plot_container, 'padding_' + side, padd)
                else:
                    padd = int(pad * 200)
                    setattr(self.plot_container, 'padding_' + side, padd)

        return panel

    def _refresh_container(self):
        ''' rebuild the container for the current data
        '''
        broadcaster = BroadcasterTool()

        mfn_line = self.value
     #   print self.panel.GetSize()
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
        y = ArrayDataSource(mfn_line.ydata, sort_order="none")

        index_range = DataRange1D()
        index_range.add(index)
        index_mapper = LinearMapper(range=index_range)

        value_range = DataRange1D(low_setting=0.0)
        value_range.add(y)
        value_mapper = LinearMapper(range=value_range)

        styles_m = list(adapter.line_style.values())
        line_style = styles_m[0]
        line_color = adapter.line_color[0]

        line_plot = self.lplot = LinePlot(index=index, value=y,
                                          index_mapper=index_mapper,
                                          value_mapper=value_mapper,
                                          color=line_color,
                                          width=25,
                                          edge_color='blue',
                                          linestyle=line_style,
                                          border_visible=False)

        add_default_grids(line_plot)
        add_default_axes(line_plot, vtitle="Y", htitle="X")

        self.plot_container.add(line_plot)
        line_plot.tools.append(PanTool(line_plot))
        line_plot.overlays.append(ZoomTool(line_plot))

#        pan = PanTool(line_plot)
#        zoom = SimpleZoom(line_plot, tool_mode="box", always_on=False)
#        broadcaster.tools.append(pan)
#        broadcaster.tools.append(zoom)
#
#        # Add the traits inspector tool to the container
#        #
#        self.plot_container.tools.append(TraitsTool( self.plot_container ))

    #--------------------------------------------------------------------------
    #  Creates the table editing tool bar:
    #--------------------------------------------------------------------------

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

    #--------------------------------------------------------------------------
    #  Handles the user requesting that columns not be sorted:
    #--------------------------------------------------------------------------
    def on_savedata(self):
        """ Handles the user requesting that 
            the data of the function is to be saved.
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

                #                factory  = self.factory
                #                plotitem = factory.plotitem
                #                x_values = getattr(self.object, plotitem.index)
                #                y_values = getattr(self.object, plotitem.value)
                x_values = self.value.xdata
                y_values = self.value.ydata
                np.savetxt(path, np.vstack((x_values, y_values)).transpose())
            except:
                print("Error saving!")
                raise
            print("Plot saved.")
        return

    def on_savefig(self):
        """ Handles the user requesting that the image of the 
            function is to be saved.
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

                """ Now we create a canvas of the appropriate size and 
                    ask it to render our component.  
                   (If we wanted to display this plot in a window, we
                    would not need to create the graphics context ourselves; 
                    it would be created for us by the window.)
                """
#                self._plot.bounds = [500,300]
#                self._plot.padding = 50
               # plot_gc = PlotGraphicsContext(self._plot.outer_bounds)
               # plot_gc.render_component(self._plot)

                #self._plot_container.outer_bounds = list((800,600))
#                plot_gc = PlotGraphicsContext((400,300), dpi=72.0)
#                plot_gc.render_component(self._plot_container)

                self.lplot.bounds = [500, 300]
                self.lplot.padding = 50

                win_size = self.lplot.outer_bounds
                plot_gc = PlotGraphicsContext(win_size)

                # Have the plot component into it
                plot_gc.render_component(self.lplot)

                # Finally, we tell the graphics context
                # to save itself to disk as an image.
                plot_gc.save(path)

            except:
                print("Error saving!")
                raise
            print("Plot saved.")
        return

#------------------------------------------------------------------------------
#  'MFnChacoEditorToolbar' class:
#------------------------------------------------------------------------------


class MFnChacoEditorToolbar (HasPrivateTraits):
    """ Toolbar displayed in table editors.
    """
    #--------------------------------------------------------------------------
    #  Trait definitions:
    #--------------------------------------------------------------------------

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
