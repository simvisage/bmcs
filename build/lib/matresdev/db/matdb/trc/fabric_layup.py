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
# Thanks for using Simvisage open source!
#
# Created on Feb 23, 2010 by: rch

from traits.api import \
    HasTraits, Directory, List, Int, Float, Any, \
    on_trait_change, File, Constant, Instance, Trait, \
    Array, Str, Property, cached_property, WeakRef, \
    Dict, Button, Bool, Enum, Event, implements, \
    DelegatesTo, Expression, Regex, Callable

from traitsui.api import \
    View, Item

# overload the 'get_label' method from 'Item' to display units in the label
from util.traits.ui.item import \
    Item

from numpy import \
    array

from .fabric_layout import \
    FabricLayOut

from matresdev.db.simdb.simdb_class import \
    SimDBClass, SimDBClassExt

from math import \
    cos, sin, pi


class FabricLayUp(SimDBClass):

    '''Describes the type and arrangement of the textile reinforcement
    In the specification of a fabric layup only one type of textile can be used.
    There is an implicit information about the thickness of the fabric layup 
    given by the spacing and the total number of layers. A constant reinforcement
    ratio is assumed (periodic fabric layup).
    '''
    #--------------------------------------------------------------------
    # register a change of the traits with metadata 'input'
    #--------------------------------------------------------------------

    input_change = Event

    @on_trait_change('+input')
    def _set_input_change(self):
        self.input_change = True

    #--------------------------------------------------------------------
    # textile fabric used
    #--------------------------------------------------------------------
    #
    fabric_layout_key = Enum(list(FabricLayOut.db.keys()),
                             simdb=True, input=True, auto_set=False, enter_set=True)

    fabric_layout_ref = Property(
        Instance(SimDBClass), depends_on='fabric_layout_key')

    @cached_property
    def _get_fabric_layout_ref(self):
        return FabricLayOut.db[self.fabric_layout_key]

    #--------------------------------------------------------------------
    # number of layers and spacing
    #--------------------------------------------------------------------

    # total number of layers
    #
    n_layers = Int(1, unit='-', input=True, table_field=True,
                   auto_set=False, enter_set=True)

    # equidistant spacing in thickness direction (z-direction) between the layers
    #
    s_tex_z = Float(1, unit='m', input=True, table_field=True,
                    auto_set=False, enter_set=True)

    #--------------------------------------------------------------------
    # orientation of the layers in the layup:
    #--------------------------------------------------------------------
    # angle with respect to the x-direction of the fabric layup and the
    # 0-degree orientation of the fabric layout. Argument 'n' represents
    # the number of an individual layer. Numbering runs in z-direction
    # from buttom to top starting with '1' (=bottom layer)

    # predefinde functions for the standard fabric layups as Callable
    #

    def get_orientation_90_0(n):
        '''return an angle alternating between 90- and 0-degree
        starting with 90-degree for n = 1 (bottom layer)
        '''
        alpha_degree = abs(sin(pi / 2 * n) * 0. + cos(pi / 2 * n) * 90.)
        return alpha_degree

    def get_orientation_all0(n):
        '''Return an constant angle = 0-degree
        for all layers.
        '''
        return 0.

    def get_orientation_all90(n):
        '''Return an constant angle = 90-degree
        for all layers.
        '''
        return 90.

    orientation_fn_dict = {
        'all0': get_orientation_all0,
        '90_0': get_orientation_90_0,
        'all90': get_orientation_all90,
        'unreinforced': get_orientation_all0
    }

    orientation_fn_key = Enum(list(orientation_fn_dict.keys()),
                              input=True, table_field=False,
                              auto_set=False, enter_set=True)

    orientation_fn_ref = Property(
        Callable, unit='degree', depends_on='input_change')

    @cached_property
    def _get_orientation_fn_ref(self):
        return self.orientation_fn_dict[self.orientation_fn_key]

    #--------------------------------------------------------------------
    # derived properties of the fabric layup
    #--------------------------------------------------------------------

    orientation_list = Property(
        List(Int, unit='degree'), depends_on='input_change')

    @cached_property
    def _get_orientation_list(self):
        return [int(self.orientation_fn_ref(i)) for i in range(self.n_layers)]

    n_layers_0 = Property(Int, unit='-', depends_on='input_change')

    @cached_property
    def _get_n_layers_0(self):
        return self.orientation_list.count(0.)

    n_layers_90 = Property(Int, unit='-', depends_on='input_change')

    @cached_property
    def _get_n_layers_90(self):
        return self.orientation_list.count(90.)

    a_tex = Property(Float, unit='mm^2/m', depends_on='input_change')

    @cached_property
    def _get_a_tex(self):
        '''total cross-sectional-area of the textile reinforcement [mm^2/m]
        for the entire FabricLayUp.
        '''
        a_tex_0 = self.fabric_layout_ref.a_tex_0  # [mm^2/m]
        a_tex_90 = self.fabric_layout_ref.a_tex_90  # [mm^2/m]
        return a_tex_0 * self.n_layers_0 + a_tex_90 * self.n_layers_90

    rho = Property(Float, unit='-', depends_on='input_change')

    @cached_property
    def _get_rho(self):
        '''reinforcement ration of the fabric layup assuming a periodic layup.
        '''
        a_tex = self.a_tex / 1000000.  # [mm^2/m]-->[m^2/m]
        return a_tex / self.thickness  # [-]

    thickness = Property(Float, unit='m', depends_on='input_change')

    @cached_property
    def _get_thickness(self):
        '''thickness of the fabric layup assuming a periodic layup.
        '''
        return self.n_layers * self.s_tex_z  # [m]

    E_tex_arr = Property(Array(Float, unit='MPa'), depends_on='input_change')

    @cached_property
    def _get_E_tex_arr(self):
        '''get the E-modulus of each layer depending on
        its orientation (usually E_tex = E_tex_0 = E_tex_90).
        '''
        return array([self.fabric_layout_ref.get_E_tex(self.orientation_fn_ref(layer_i))
                      for layer_i in range(self.n_layers)])

    E_tex = Property(Float, unit='MPa', depends_on='input_change')

    @cached_property
    def _get_E_tex(self):
        '''get the average E-modulus of all layers, i.e. the 
        entire fabric layup (usually E_tex = E_tex_0 = E_tex_90).
        '''
        return sum(self.E_tex_arr) / self.n_layers

    a_roving_0 = DelegatesTo('fabric_layout_ref')
    a_roving_0 = DelegatesTo('fabric_layout_ref')
    #------------------------------------------------------------------
    # layup view:
    #------------------------------------------------------------------

    traits_view = View(
        Item('fabric_layout_key'),
        Item('n_layers', format_str="%.0f"),
        Item('s_tex_z', format_str="%.5f"),
        Item('orientation_fn_key'),
        resizable=True,
        scrollable=True,
        height=0.4,
        width=0.5,
    )

#------------------------------------------------------------------

if __name__ == '__main__':
    flu = FabricLayUp(
        n_layers=7,
        s_tex_z=0.030 / (7 + 1),
        orientation_fn_key='all90',
        fabric_layout_key='MAG-07-03'
    )
    flu.configure_traits()
