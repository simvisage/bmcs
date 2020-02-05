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

from traits.api import \
    HasTraits, Directory, List, Int, Float, Any, \
    on_trait_change, File, Constant, Instance, Trait, \
    Array, Str, Property, cached_property, WeakRef, \
    Dict, Button, Bool, Enum, Event, implements, \
    DelegatesTo

from traitsui.api import \
    View, Item, DirectoryEditor, TabularEditor, HSplit, VGroup, \
    TableEditor, EnumEditor, Handler, FileEditor, VSplit, Group, \
    InstanceEditor, HGroup, Spring

# # overload the 'get_label' method from 'Item' to display units in the label
from util.traits.ui.item import \
    Item

from matresdev.db.simdb.simdb_class import \
    SimDBClass, SimDBClassExt

from math import \
    cos, sin, pi

class FabricLayOut(SimDBClass):
    '''Comprises the characteristics of the textile reinforcement
    '''
    # cross sectional area of the reinforcement [mm^2/m]:
    # in 0 and 90 degree orientation:
    a_tex_0 = Float(unit='mm^2/m', simdb=True)
    a_tex_90 = Float(unit='mm^2/m', simdb=True)
    E_tex_0 = Float(unit='MPa' , simdb=True)
    E_tex_90 = Float(unit='MPa' , simdb=True)

    # spacing of the textile mesh [mm]:
    s_tex_0 = Float(unit='mm', simdb=True)
    s_tex_90 = Float(unit='mm', simdb=True)

    def get_E_tex(self, angle_degree):
        angle_rad = pi / 180. * angle_degree
        return self.E_tex_0 * cos(angle_rad) + self.E_tex_90 * sin(angle_rad)

    a_roving_0 = Property(Float, depends_on='a_tex_0, s_tex_0')
    @cached_property
    def _get_a_roving_0(self):
        n_rovings_in_m = 1000.0 / self.s_tex_0
        return self.a_tex_0 / n_rovings_in_m
        
    a_roving_90 = Property(Float, depends_on='a_tex_90, s_tex_90')
    @cached_property
    def _get_a_roving_90(self):
        n_rovings_in_m = 1000.0 / self.s_tex_90
        return self.a_tex_90 / n_rovings_in_m
        
    # view:
    traits_view = View(
                      Item('key'     , style='readonly'),
                      Item('a_tex_0' , style='readonly', format_str="%.2f"),
                      Item('a_tex_90', style='readonly', format_str="%.2f"),
                      Item('E_tex_0' , style='readonly', format_str="%.0f"),
                      Item('E_tex_90', style='readonly', format_str="%.0f"),
                      Item('s_tex_0' , style='readonly', format_str="%.1f"),
                      Item('s_tex_90', style='readonly', format_str="%.1f"),
                      Item('a_roving_0' , style='readonly', format_str="%.1f"),
                      Item('a_roving_90', style='readonly', format_str="%.1f"),
                      resizable=True,
                      scrollable=True
                      )

# Setup the database class extension
#
FabricLayOut.db = SimDBClassExt(
            klass=FabricLayOut,

            constants={
               'unreinforced' : FabricLayOut(
                                           a_tex_0=0.,
                                           a_tex_90=0.,
                                           E_tex_0=0.,
                                           E_tex_90=0.,
                                           s_tex_0=1.,
                                           s_tex_90=1.,
                                           ),

               # AR-glas textile (2400 tex) compact binding (Franse):
               # new textile tag "2D-03-08" corresponds to "MAG-07-03"
               #
               'MAG-07-03' : FabricLayOut(
                                           a_tex_0=107.89,
                                           a_tex_90=106.61,
#                                           E_tex_0=72000.,
#                                           E_tex_90=72000.,
                                           E_tex_0=66831.,  # (l=500mm;2400tex;200mm/min)
                                           E_tex_90=66831.,  # (l=500mm;2400tex;200mm/min)
                                           s_tex_0=8.3,
                                           s_tex_90=8.4,
                                           ),

               # AR-glas textile (2400 tex) tricot binding:
               #
               '2D-15-10' : FabricLayOut(
                                           a_tex_0=107.89,
                                           a_tex_90=106.61,
#                                           E_tex_0=72000.,
#                                           E_tex_90=72000.,
                                           E_tex_0=66831.,  # (l=500mm;2400tex;200mm/min)
                                           E_tex_90=66831.,  # (l=500mm;2400tex;200mm/min)
                                           s_tex_0=8.3,
                                           s_tex_90=8.4,
                                           ),


               # AR-glas textile (2 x 1200 tex in 0-direction):
               #
               '2D-02-06a' : FabricLayOut(
                                           a_tex_0=71.65,
                                           a_tex_90=53.31,
#                                           E_tex_0=72000.,
#                                           E_tex_90=72000.,
                                           E_tex_0=66831.,  # (l=500mm;2400tex;200mm/min)
                                           E_tex_90=66831.,  # (l=500mm;2400tex;200mm/min)
                                           s_tex_0=12.5,
                                           s_tex_90=8.4,
                                           ),

               # carbon textile / tricot binding ("Trikot")
               # 2 x 800 tex in 0-direction (2v1l): spacing 12,5 mm
               # 1 x 800 tex in 90-direction (1v1l): effective spacing of 7.7 mm!
               #
               '2D-14-10' : FabricLayOut(
                                           a_tex_0=73.89,
                                           a_tex_90=58.,  # 53.9 * 8.4mm / 7.7mm
                                           E_tex_0=180862.,  # stiffness value taken from yarn tests 1600tex, l=300mmm;
                                           E_tex_90=180862.,  # stiffness value taken from yarn tests 1600tex, l=300mmm;
                                           s_tex_0=12.5,
                                           s_tex_90=5.83,  # input for manufacturing machine was 8.4 mm!
                                           ),

               # carbon textile / tricot binding ("Trikot")
               # 1 x 800 tex in 0-direction (1v1l): 8.3 mm
               # 1 x 800 tex in 90-direction (1v1l): effective spacing of 7.7 mm!
               #
               '2D-04-11' : FabricLayOut(
                                           a_tex_0=53.9,
                                           a_tex_90=58.,  # 53.9 * 8.4mm / 7.7mm
                                           E_tex_0=180862.,  # stiffness value taken from yarn tests 1600tex, l=300mmm;# yarn tests with l=125mm: E=165GPa
                                           E_tex_90=180862.,  # stiffness value taken from yarn tests 1600tex, l=300mmm;
                                           s_tex_0=8.3,
                                           s_tex_90=7.7,  # input for manufacturing machine was 8.4 mm!
                                           ),

               # carbon textile / tricot binding ("Trikot") with defect
               # due to the manifacturing process (rovings are separated by the binding thread)
               # 1 x 800 tex in 0-direction (1v1l): 8.3 mm
               # 1 x 800 tex in 90-direction (1v1l): effective spacing of 7.7 mm!
               #
               '2D-04-11_defect' : FabricLayOut(
                                           a_tex_0=53.9,
                                           a_tex_90=58.,  # 53.9 * 8.4mm / 7.7mm
                                           E_tex_0=180862.,  # stiffness value taken from yarn tests 1600tex, l=300mmm;# yarn tests with l=125mm: E=165GPa
                                           E_tex_90=180862.,  # stiffness value taken from yarn tests 1600tex, l=300mmm;
                                           s_tex_0=8.3,
                                           s_tex_90=7.7,  # input for manufacturing machine was 8.4 mm!
                                           ),

               # demonstrator textile (carbon)
               # carbon textile / tissue binding ("Tuch")
               # 1 x 800 tex in 0-direction (1v1l): 8.3 mm
               # 1 x 800 tex in 90-direction (1v1l): effective spacing of 7.7 mm!
               #
               '2D-05-11' : FabricLayOut(
                                           a_tex_0=53.9,
#                                           a_tex_0 = 55.4,
                                           a_tex_90=58.,  # 53.9 * 8.4mm / 7.7mm
#                                           a_tex_90=60.,  # 55.0 * 8.4mm / 7.7mm
                                           E_tex_0=180862.,  # stiffness value taken from yarn tests 1600tex, l=300mmm;# yarn tests with l=125mm: E=165GPa
                                           E_tex_90=180862.,  # stiffness value taken from yarn tests 1600tex, l=300mmm;
                                           s_tex_0=8.3,
                                           s_tex_90=7.7,  # input for manufacturing machine was 8.4 mm!
                                           ),

               # AR-glass tissue binding (barrel shell)
               # tissue binding ("Tuch")
               # 1 x 1200 tex in 0-direction (1v1l): 8.3 mm
               # 1 x 1200 tex in 90-direction (1v1l): effective spacing of 7.7 mm!
               #
               '2D-09-12' : FabricLayOut(
                                           a_tex_0=54.0,
#                                           a_tex_0=55.4,
                                           a_tex_90=58.2,  # 0.448mm2 / 7.7mm
#                                           a_tex_90=60.,  # 55.0 * 8.4mm / 7.7mm
#                                           E_tex_0=72000.,
#                                           E_tex_90=72000.,
                                           E_tex_0=66831.,  # (l=500mm;2400tex;200mm/min)
                                           E_tex_90=66831.,  # (l=500mm;2400tex;200mm/min)
                                           s_tex_0=8.3,
                                           s_tex_90=7.7,  # input for manufacturing machine was 8.4 mm!
                                           ),


               # carbon textile (heavy tow 3300 tex): Trikot binding
               # in SFB-yarn tensile test: sig_max = 1020 MPa, eps_max = 10,3E-3
               #
               '2D-18-10' : FabricLayOut(
                                           a_tex_0=76.96,
                                           a_tex_90=123,
                                           E_tex_0=180862.,  # stiffness value taken from yarn tests 1600tex, l=300mmm; # yarn tests with l=125mm: E=107500 MPa
                                           E_tex_90=180862.,
                                           s_tex_0=24.0,
                                           s_tex_90=15.0,
                                           ),

               # carbon with epoxid rasin:
               # 3300 tex
               #
               'C-Grid-C50' : FabricLayOut(
                                           a_tex_0=40.0,
                                           a_tex_90=44.9,
                                           E_tex_0=234500.,
                                           E_tex_90=234500.,
                                           s_tex_0=46.0,
                                           s_tex_90=41.0,
                                           ),

               # EP coated carbon textile
               # 2 x 24 K in 0-direction
               # 2 x 24 K in 90-direction
               #
               'C-Grid-C50-25' : FabricLayOut(
                                           a_tex_0=74.,
                                           a_tex_90=74.,
                                           E_tex_0=234500.,
                                           E_tex_90=234500.,
                                           s_tex_0=2.5,
                                           s_tex_90=2.5,
                                           ),

               # SBR coated carbon textile
               # 1 x 50K in 0-direction
               # 1 x 50K in 90-direction
               #
               'Grid-600' : FabricLayOut(
                                           a_tex_0=170.,
                                           a_tex_90=102.,
                                           E_tex_0=180000.,  # stiffness value taken from yarn tests #165000.,
                                           E_tex_90=180000.,
                                           s_tex_0=10.8,
                                           s_tex_90=18.,
                                           ),

               # SBR coated carbon textile
               # 2 x 24K in 0-direction
               # 2 x 24K in 90-direction
               #
               'FRA-CAR/SB' : FabricLayOut(
                                           a_tex_0=55.,
                                           a_tex_90=46.,
                                           E_tex_0=1.,
                                           E_tex_90=1.,
                                           s_tex_0=2.3,
                                           s_tex_90=2.1,
                                           ),

               # EP coated AR-glas textile
               #
               'FRA-AR/EP' : FabricLayOut(
                                           a_tex_0=107.,
                                           a_tex_90=134.,
                                           E_tex_0=1.,
                                           E_tex_90=1.,
                                           s_tex_0=2.0,
                                           s_tex_90=1.6,  # average spacing for alternating arrangement of rovings (3/1/1)
                                           ),

               # SBR coated carbon textile
               # 1 x 12K (800tex) in 0-direction
               # ? x ? K in 90-direction
               #
               'CAR-800-SBR_TUD' : FabricLayOut(
                                           a_tex_0=62.8,
                                           a_tex_90=1.,
                                           E_tex_0=245000.,  # SBR coating 800 tex
                                           E_tex_90=245000.,
                                           s_tex_0=7.1,
                                           s_tex_90=1.,
                                           ),

               # SBR coated carbon textile
               # 12K (800tex) rovings with SBR-coating
               # A_rov = 0.45m^2
               #
               'NWM3-016-09-b1' : FabricLayOut(
                                           a_tex_0=62.5,  # = 0.45 m^2 / 0.0072 m
                                           a_tex_90=31.3,  # = 0.45 m^2 / 0.0144 m
                                           E_tex_0=245000.,  # SBR coating 800 tex
                                           E_tex_90=245000.,
                                           s_tex_0=7.2,
                                           s_tex_90=14.4,
                                           ),

               # SBR coated carbon textile
               # 50K (3300tex) rovings with SBR-coating 0-direction
               # 12L (800tex) rovings with SBR-coating in 90-direction
               # A_rov = 1.84m^2
               #
               'CAR-3300-SBR_BTZ2' : FabricLayOut(
                                           a_tex_0=144.9,  # = 1.84 m^2 / 0.0127 m
                                           a_tex_90=25.0,  # = 0.45 m^2 / 0.018 m
                                           E_tex_0=170000.,  # yarn test with SBR coating (3300tex)
                                           E_tex_90=152000.,  # yarn test with SBR coating (800tex)
                                           s_tex_0=12.7,
                                           s_tex_90=18.0,
                                           ),

               # epoxy resin coated carbon textile
               # 50K (3300tex) rovings with EP-coating 0-direction and 90-direction
               # A_rov = 1.84m^2
               #
               'CAR-3300-EP_Q90' : FabricLayOut(
                                           a_tex_0=90.,  # = 1.84 m^2 / 0.02 m
                                           a_tex_90=90,
                                           E_tex_0=245000.,  # yarn test with EP impregnation (3300tex)
                                           E_tex_90=245000.,
                                           s_tex_0=21.,
                                           s_tex_90=21.,
                                           ),
#
               'Q85/85-CCE-21' : FabricLayOut(
                                           a_tex_0=85.,  # = 1.81 m^2 / 0.021 m
                                           a_tex_90=85,
                                           E_tex_0=246100.,  # Modulus from single yarn test data base
                                           E_tex_90=246100., 
                                           s_tex_0=21.,
                                           s_tex_90=21.,
                                           ),
#
               'Q95/95-CCE-38' : FabricLayOut(
                                           a_tex_0=95.,  # = 3.62 m^2 / 0.038 m
                                           a_tex_90=95,
                                           E_tex_0=226474.,  # Modulus from single yarn test data base
                                           E_tex_90=236362., 
                                           s_tex_0=38.,
                                           s_tex_90=38.,
                                           ),  
#
               'Q142/142-CCE-25' : FabricLayOut(
                                           a_tex_0=142,  # = 3.62 m^2 / 0.025 m
                                           a_tex_90=142,
                                           E_tex_0=246100.,  # Modulus from single yarn test data base
                                           E_tex_90=246100., 
                                           s_tex_0=25.,
                                           s_tex_90=25.,
                                           ), 
#
               'Q142/142-CCE-38' : FabricLayOut(
                                           a_tex_0=142,  # = 5.42 m^2 / 0.038 m
                                           a_tex_90=142,
                                           E_tex_0=246100.,  # Modulus from single yarn test data base
                                           E_tex_90=246100., 
                                           s_tex_0=38.,
                                           s_tex_90=38.,
                                           ), 
#
               'Q145/145-AAE-25' : FabricLayOut(
                                           a_tex_0=145,  # = 3.69 m^2 / 0.025 m
                                           a_tex_90=145,
                                           E_tex_0=65000.,  # ToDo: Verify Modulus by comparison to roving tests
                                           E_tex_90=65000., # ToDo: Verify Modulus by comparison to roving tests
                                           s_tex_0=25.,
                                           s_tex_90=25.,
                                           ), 
#
               'Q121/121-AAE-38' : FabricLayOut(
                                           a_tex_0=121,  # = 4.62 m^2 / 0.038 m
                                           a_tex_90=121,
                                           E_tex_0=65000.,  # ToDo: Verify Modulus by comparison to roving tests
                                           E_tex_90=65000., # ToDo: Verify Modulus by comparison to roving tests
                                           s_tex_0=38.,
                                           s_tex_90=38.,
                                           ),    
#
               'Q97/97-AAE-38' : FabricLayOut(
                                           a_tex_0=97,  # = 3.69 m^2 / 0.038 m
                                           a_tex_90=97,
                                           E_tex_0=65000.,  # ToDo: Verify Modulus by comparison to roving tests
                                           E_tex_90=65000., # ToDo: Verify Modulus by comparison to roving tests
                                           s_tex_0=38.,
                                           s_tex_90=38.,
                                           ),
#
               'Q87/87-AAE-21' : FabricLayOut(
                                           a_tex_0=87,  # = 1.85 m^2 / 0.025 m
                                           a_tex_90=87,
                                           E_tex_0=65000.,  # ToDo: Verify Modulus by comparison to roving tests
                                           E_tex_90=65000., # ToDo: Verify Modulus by comparison to roving tests
                                           s_tex_0=21.,
                                           s_tex_90=21.,
                                           ),
#
               'Q87/87-AAS-21' : FabricLayOut(
                                           a_tex_0=87,  # = 1.85 m^2 / 0.025 m
                                           a_tex_90=87,
                                           E_tex_0=65000.,  # ToDo: Verify Modulus by comparison to roving tests
                                           E_tex_90=65000., # ToDo: Verify Modulus by comparison to roving tests
                                           s_tex_0=21.,
                                           s_tex_90=21.,
                                           ), 
#
               'Q142/142-CCS-25' : FabricLayOut(
                                           a_tex_0=142,  # = 3.62 m^2 / 0.025 m
                                           a_tex_90=142,
                                           E_tex_0=65000.,  # ToDo: Verify Modulus by comparison to roving tests
                                           E_tex_90=65000., # ToDo: Verify Modulus by comparison to roving tests
                                           s_tex_0=25.,
                                           s_tex_90=25.,
                                           ),
#
               'R106/29-CGS-17x31' : FabricLayOut(
                                           a_tex_0=106,  # = 1.81 m^2 / 0.017 m
                                           a_tex_90=29,
                                           E_tex_0=245000.,  # ToDo: Verify Modulus by comparison to roving tests
                                           E_tex_90=65000., # ToDo: Verify Modulus by comparison to roving tests
                                           s_tex_0=17.,
                                           s_tex_90=31.,
                                           ),
#                            
               'CAR-6600-SBR_E0003' : FabricLayOut(
                                           a_tex_0=141.,  # = 3.67 m^2 / 0.026 m
                                           a_tex_90=141,
                                           E_tex_0=245000.,  # ToDo: Verify Modulus by comparison to roving tests
                                           E_tex_90=245000., # ToDo: Verify Modulus by comparison to roving tests
                                           s_tex_0=26.,
                                           s_tex_90=26.,
                                           ),    
               # styrol-butadiene impregnated Textile, Manufacturer: V.Fraas
               # 100K (6600tex) rovings in 0-direction and 90-direction
               # A_rov = 3.67mm^2
               #                                                                                                                              
             }
            )

if __name__ == '__main__':
    FabricLayOut.db.configure_traits()
