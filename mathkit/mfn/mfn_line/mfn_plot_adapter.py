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

from traits.api import \
    HasTraits, HasPrivateTraits, Color, Str, Int, Float, Enum, List, \
    Dict, Bool, Instance, Any, Font, Event, Property, Interface, \
    on_trait_change, cached_property, Color, Tuple
           
#-------------------------------------------------------------------------------
#  'MFnPlotAdapter' class:
#-------------------------------------------------------------------------------


class MFnPlotAdapter (HasPrivateTraits):
    
    """ The base class for adapting function implementation in order to plot
    it in a plotting toolkit (either Chaco or Matplotlib)
    """
    
    # label on the x axis (only used when var_x empty)
    #
    label_x = Str('')
    
    # label on the y axis (only used when var_y empty)
    label_y = Str('')

    # title of the diagram
    #
    title = Str('')
    
    # color of the title
    title_color = Str('black')
    
    # when specified take the label of the axis from the value 
    # of the variable trait in object
    #
    var_x = Str('')
    
    # when specified take the label of the axis from the value 
    # of the variable trait in object
    #
    var_y = Str('')
    
    # label for line in legend
    #
    line_label = Str('')
    
    # @todo - further attributes to be made available
    
    # limits of number positions for switching into scientific notation 1eXX
    scilimits = Tuple(-3., 4.)
    
    # Plot properties
    
    line_color = List(['blue',
                        'red',
                        'green',
                        'black',
                        'magenta',
                        'yellow',
                        'white',
                        'cyan'
                        ]) 

    # rr: is the dictionary necessary here?
    # why not a list of keys? chaco as well mpl can work with the key 
    line_style = Dict({'solid' : '-',
                  'dash' : '.',
                  'dot dash' : '-.',
                  'dot' : ':',
                 })

    # horisontal axis scale ('log', 'symlog', 'linear')    
    xscale = Str('linear')
    
    # vertical axis scale ('log', 'symlog', 'linear')    
    yscale = Str('linear')    
    
    # linewidth
    line_width = List(2.0)
    
    # color of the plotting background
    bgcolor = Str('white')
  
    # Border, padding properties
    border_visible = Bool(False)

    # @todo - unused - padding defined below - should be used in Chaco - 
    # in Matplotlib it is working already [Faezeh, Check please]
    border_width = Int(0)
    
    # @todo does this work in both Mpl and Chaco? [Faeseh Check please]
    #
    padding_bg_color = Str('white')
    
    # labels in legend
    
    # @todo - no labels here - the defaults should make sense - are they active?
    # 
    legend_labels = Tuple()
    
    # maximum size - None means arbitrarily resizable
    #
    max_size = Tuple()
    
    # minimum size
    #
    min_size = Tuple((200, 200))

    # padding
    
    padding = Dict({ 'left'   : 0.1,
                      'bottom' : 0.1,
                      'right'  : 0.9,
                      'top'    : 0.9 })
    
    # No of values displayed on the x axis
    xticks = Int(0)
    
    # No of values displayed on the y axis
    yticks = Int(0)

#-------------------------------------------------------------------------------
#  'MFnPlotAdapter' class:
#-------------------------------------------------------------------------------


class MFnMultiPlotAdapter (MFnPlotAdapter):
    
    """ The base class for adapting function implementation in order to plot
    it in a plotting toolkit (either Chaco or Matplotlib)
    """

    # rr: test only, needs to be written in a more robust way
    mline_style = List(100 * ['solid'])
    
    mline_color = List(100 * ['black'])
    
    mline_width = List(100 * [2.0])
    
    
