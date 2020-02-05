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
# Created on Feb 23, 2010 by: rch

from enthought.traits.api import \
    HasTraits, Directory, List, Int, Float, Any, \
    on_trait_change, File, Constant, Instance, Trait, \
    Array, Str, Property, cached_property, WeakRef, \
    Dict, Button, Bool, Enum, Event, implements, \
    DelegatesTo

from enthought.traits.ui.api import \
    View, Item, DirectoryEditor, TabularEditor, HSplit, VGroup, \
    TableEditor, EnumEditor, Handler, FileEditor, VSplit, Group, \
    InstanceEditor, HGroup, Spring

## overload the 'get_label' method from 'Item' to display units in the label
from util.traits.ui.item import \
    Item

from promod.simdb.simdb_class import \
    SimDBClass, SimDBClassExt

from math import \
    cos, sin, pi

class FabricLayOut( SimDBClass ):
    '''Comprises the characteristics of the textile reinforcement
    '''
    # cross sectional area of the reinforcement [mm^2/m]: 
    # in 0 and 90 degree orientation: 
    a_tex_0 = Float( unit = 'mm^2/m', simdb = True )
    a_tex_90 = Float( unit = 'mm^2/m', simdb = True )
    E_tex_0 = Float( unit = 'MPa' , simdb = True )
    E_tex_90 = Float( unit = 'MPa' , simdb = True )

    # spacing of the textile mesh [mm]: 
    s_tex_0 = Float( unit = 'mm', simdb = True )
    s_tex_90 = Float( unit = 'mm', simdb = True )

    def get_E_tex( self, angle_degree ):
        angle_rad = pi / 180. * angle_degree
        return self.E_tex_0 * cos( angle_rad ) + self.E_tex_90 * sin( angle_rad )

    # view:
    traits_view = View( 
                      Item( 'key'     , style = 'readonly' ),
                      Item( 'a_tex_0' , style = 'readonly', format_str = "%.2f" ),
                      Item( 'a_tex_90', style = 'readonly', format_str = "%.2f" ),
                      Item( 'E_tex_0' , style = 'readonly', format_str = "%.0f" ),
                      Item( 'E_tex_90', style = 'readonly', format_str = "%.0f" ),
                      Item( 's_tex_0' , style = 'readonly', format_str = "%.1f" ),
                      Item( 's_tex_90', style = 'readonly', format_str = "%.1f" ),
                      resizable = True,
                      scrollable = True
                      )

# Setup the database class extension 
#
FabricLayOut.db = SimDBClassExt( 
            klass = FabricLayOut,

            constants = {
               'unreinforced' : FabricLayOut( 
                                           a_tex_0 = 0.,
                                           a_tex_90 = 0.,
                                           E_tex_0 = 0.,
                                           E_tex_90 = 0.,
                                           s_tex_0 = 1.,
                                           s_tex_90 = 1.,
                                           ),

               # AR-glas textile (2400 tex):                            
               #
               'MAG-07-03' : FabricLayOut( 
                                           a_tex_0 = 107.89,
                                           a_tex_90 = 106.61,
                                           E_tex_0 = 70000,
                                           E_tex_90 = 70000,
                                           s_tex_0 = 8.3,
                                           s_tex_90 = 8.4,
                                           ),

               # AR-glas textile (2 x 800 tex in 0-direction):                            
               #
               '2D-02-06a' : FabricLayOut( 
                                           a_tex_0 = 71.65,
                                           a_tex_90 = 53.31,
                                           E_tex_0 = 70000,
                                           E_tex_90 = 70000,
                                           s_tex_0 = 12.5,
                                           s_tex_90 = 8.4,
                                           ),

               # carbon textile / tricot binding ("Trikot") 
               # 2 x 800 tex in 0-direction (2v1l): spacing 12,5 mm                            
               # 1 x 800 tex in 90-direction (1v1l): effective spacing of 7.6 mm!                           
               #
               '2D-14-10' : FabricLayOut( 
                                           a_tex_0 = 73.89,
                                           a_tex_90 = 60., #55.0 * 8.4mm / 7.7mm
                                           E_tex_0 = 165000,
                                           E_tex_90 = 165000,
                                           s_tex_0 = 12.5,
                                           s_tex_90 = 5.83, # input for manufacturing machine was 8.4 mm!
                                           ),

               # carbon textile / tricot binding ("Trikot") 
               # 1 x 800 tex in 0-direction (1v1l): 8.3 mm                            
               # 1 x 800 tex in 90-direction (1v1l): effective spacing of 7.6 mm!                            
               #
               '2D-04-11' : FabricLayOut( 
                                           a_tex_0 = 55.4,
                                           a_tex_90 = 60., #55.0 * 8.4mm / 7.7mm
                                           E_tex_0 = 165000,
                                           E_tex_90 = 165000,
                                           s_tex_0 = 8.3,
                                           s_tex_90 = 7.6, # input for manufacturing machine was 8.4 mm!
                                           ),
               # demonstrator textile
               # carbon textile / tissue binding ("Tuch") 
               # 1 x 800 tex in 0-direction (1v1l): 8.3 mm                            
               # 1 x 800 tex in 90-direction (1v1l): effective spacing of 7.6 mm!                            
               #
               '2D-05-11' : FabricLayOut( 
                                           a_tex_0 = 55.4,
                                           a_tex_90 = 60., #55.0 * 8.4mm / 7.7mm
                                           E_tex_0 = 165000,
                                           E_tex_90 = 165000,
                                           s_tex_0 = 8.3,
                                           s_tex_90 = 7.6, # input for manufacturing machine was 8.4 mm!
                                           ),


               # carbon textile (heavy tow 3300 tex): Trikot binding       
               # in SFB-yarn tensile test: sig_max = 1020 MPa, eps_max = 10,3E-3                     
               #
               '2D-18-10' : FabricLayOut( 
                                           a_tex_0 = 76.96,
                                           a_tex_90 = 110,
                                           E_tex_0 = 107500,
                                           E_tex_90 = 107500,
                                           s_tex_0 = 25.0,
                                           s_tex_90 = 16.8,
                                           ),

               # carbon with epoxid rasin:                            
               #
               'C-Grid-C50' : FabricLayOut( 
                                           a_tex_0 = 37.53,
                                           a_tex_90 = 42.03,
                                           E_tex_0 = 234500,
                                           E_tex_90 = 234500,
                                           s_tex_0 = 46.0,
                                           s_tex_90 = 41.0,
                                           )
             }
            )

if __name__ == '__main__':
    FabricLayOut.db.configure_traits()
