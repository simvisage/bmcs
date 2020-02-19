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
# Thanks for using Simvisage open source!
#
# Created on Jan 26, 2010 by: rch

from traits.api import \
    HasTraits, Float, Property, cached_property, \
    Instance, List, on_trait_change, Int, Tuple, Bool, \
    DelegatesTo, Event, Str, Button, Dict, Array, Any, \
    implements

from traitsui.api import \
    View, Item, Tabbed, VGroup, HGroup, ModelView, HSplit, VSplit, \
    CheckListEditor, EnumEditor, TableEditor, TabularEditor, Handler

class ToDo( HasTraits ):
    todo_string = Str( 'This is on a todo list' )
    traits_view = View( Item( 'todo_string', show_label = False, springy = True,
                              style = 'readonly' ),
                        width = 0.2,
                        height = 0.2,
                        buttons = ['OK'] )


