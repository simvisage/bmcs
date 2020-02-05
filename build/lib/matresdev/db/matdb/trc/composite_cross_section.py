# -------------------------------------------------------------------------
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
    List, Int, Float, \
    on_trait_change, Instance, \
    Array, Property, cached_property, \
    Bool, Enum, Event

from traitsui.api import \
    View, Item, HSplit, VGroup, \
    TableEditor, VSplit, Group, \
    Spring

from traitsui.table_column import \
    ObjectColumn

from numpy import \
    array, ones_like, argmax

import numpy as np

from .concrete_mixture \
    import ConcreteMixture

from .fabric_layup \
    import FabricLayUp

from matresdev.db.simdb.simdb_class import \
    SimDBClass

from .ccs_unit_cell import \
    CCSUnitCell

ccs_table_editor = TableEditor(
    columns=[ObjectColumn(name='fabric_layout_key',
                          editable=True, width=0.20),
             ObjectColumn(name='n_layers',
                          editable=True, width=0.20),
             ObjectColumn(name='s_tex_z',
                          editable=True, width=0.20),
             ObjectColumn(name='orientation_fn_key',
                          editable=True, width=0.20),
             ObjectColumn(name='n_layers_0',
                          editable=False, width=0.20),
             ObjectColumn(name='n_layers_90',
                          editable=False, width=0.20),
             ObjectColumn(name='thickness',
                          editable=False, width=0.20),
             ObjectColumn(name='is_regular',
                          editable=False, width=0.14), ],
    selection_mode='row',
    selected='object.selected_layup',
    show_toolbar=True,
    row_factory=FabricLayUp,
    editable=True,
    auto_add=False,
    configurable=True,
    sortable=True,
    deletable=True,
    reorderable=True,
    sort_model=False,
    auto_size=False,
    #                        filters      = [EvalFilterTemplate,
    #                                       MenuFilterTemplate,
    #                                       RuleFilterTemplate ],
    #                        search       = EvalTableFilter(),
)


class CompositeCrossSection(SimDBClass):

    '''Describes the combination of the concrete mixture and textile
    cross section.
    Note: only one concrete can be defined for the entire composite
    cross section.
    '''

    input_change = Event

    @on_trait_change('+input,fabric_layup_list.input_change')
    def _set_input_change(self):
        self.input_change = True

    # -------------------------------------------------------------------------
    # select the concrete mixture from the concrete database:
    # -------------------------------------------------------------------------

    concrete_mixture_key = Enum(list(ConcreteMixture.db.keys()),
                                simdb=True, input=True,
                                auto_set=False, enter_set=True)

    concrete_mixture_ref = Property(Instance(SimDBClass),
                                    depends_on='concrete_mixture_key')

    @cached_property
    def _get_concrete_mixture_ref(self):
        return ConcreteMixture.db[self.concrete_mixture_key]

    # -------------------------------------------------------------------------
    # define the composite cross section as a list of fabric layups:
    # -------------------------------------------------------------------------

    fabric_layup_list = List(Instance(FabricLayUp), input=True)

    # default layup necessary to show view the first time.
    def _fabric_layup_list_default(self):
        return [plain_concrete(1.0)]

    selected_layup = Instance(FabricLayUp, transient=True)

    def _selected_layup_default(self):
        return self.fabric_layup_list[0]

    # -------------------------------------------------------------------------
    # calculated material properties for the composite
    # -------------------------------------------------------------------------

    thickness_arr = Property(Array(Float, unit='m'), depends_on='input_change')

    @cached_property
    def _get_thickness_arr(self):
        '''thickness of each fabric layup.
        '''
        return array([flu.thickness for flu in self.fabric_layup_list],
                     dtype='float_')

    thickness = Property(Float, unit='m', depends_on='input_change')

    @cached_property
    def _get_thickness(self):
        '''thickness of the entire composite cross section.
        '''
        return sum(self.thickness_arr)

    a_tex_arr = Property(Array(Float, unit='m'), depends_on='input_change')

    @cached_property
    def _get_a_tex_arr(self):
        '''textile cross section of each fabric layup.
        '''
        return array([flu.a_tex for flu in self.fabric_layup_list],
                     dtype='float_')

    a_tex = Property(Float, unit='mm^2/m', depends_on='input_change')

    @cached_property
    def _get_a_tex(self):
        '''textile cross section of the entire cross section per meter width.
        '''
        return sum(self.a_tex_arr)

    rho_c = Property(Float, unit='-', depends_on='input_change')

    @cached_property
    def _get_rho_c(self):
        '''reinforcement ratio of the composite material.
        '''
        return self.a_tex / self.thickness / 1000000  # [-]

    rho_arr = Property(Array(Float, unit='-'), depends_on='input_change')

    @cached_property
    def _get_rho_arr(self):
        '''reinforcement ration of each fabric layup.
        '''
        return array([flu.rho for flu in self.fabric_layup_list],
                     dtype='float_')

    E_tex_arr = Property(Array(Float, unit='MPa'), depends_on='input_change')

    @cached_property
    def _get_E_tex_arr(self):
        '''textile E-modulus of each fabric layup.
        '''
        return array([flu.E_tex for flu in self.fabric_layup_list],
                     dtype='float_')

    E_tex = Property(Float, unit='MPa', depends_on='input_change')

    @cached_property
    def _get_E_tex(self):
        '''get the larges value of the textile reinforcement
        for the plot of the analytical stiffness
        in 'exp_browse'
        '''
        return np.max(self.E_tex_arr, axis=0)
        # smeared textile E-modulus of the entire composite.
#        return sum( self.E_tex_arr * self.thickness_arr ) / self.thickness

    def get_E_m_time(self, age):
        '''function for the concrete matrix E-modulus
        as function depending of the concrete age.
        '''
        E_m = self.concrete_mixture_ref.get_E_m_time(age)
        return E_m

    def get_E_c_time(self, age):
        '''function for the composite E-modulus as weighted sum of all
        fabric layups. Returns a function depending of the concrete age.
        '''
        rho_arr = self.rho_arr
        E_tex_arr = self.E_tex_arr
        E_m = self.concrete_mixture_ref.get_E_m_time(age)
        E_m_arr = E_m * ones_like(rho_arr)
        return sum(((1 - rho_arr) * E_m_arr + rho_arr * E_tex_arr) *
                   self.thickness_arr) / self.thickness

    E_c28 = Property(Float, unit='MPa', depends_on='input_change')

    @cached_property
    def _get_E_c28(self):
        '''Composite E-modulus after 28 days.
        '''
        return self.get_E_c_time(28)

    max_rho_idx = Property(Int, unit='-', depends_on='input_change')

    @cached_property
    def _get_max_rho_idx(self):
        '''return the index of the dominating layup in 'flu_list'.
        '''
        return argmax(self.rho_arr)

    max_rho = Property(Float, unit='m', depends_on='input_change')

    @cached_property
    def _get_max_rho(self):
        '''return the reinforcement ratio of the dominating layup in 'flu_list'.
        '''
        return self.rho_arr[self.max_rho_idx]

    is_regular = Property(Bool)

    def _get_is_regular(self):
        # first check to see if this calibration is acceptable at all.
        # there must be only one layup with unreinforced coverage
        # to derive material parameters that can be reused for non-constant
        # cross sectional stress profiles
        #
        if len(self.fabric_layup_list) > 3:
            return False

        # get the proportion of the reinforcement ratio
        # of the other layers
        #
        flu_rho_fraction = self.max_rho / self.rho_c
        if (flu_rho_fraction) < 0.90:
            return False

        return True

    # -------------------------------------------------------------------------
    # Calibration parameters
    # -------------------------------------------------------------------------
    def set_param(self, material_model, calibration_test, df):

        if not self.is_regular:
            raise ValueError('Too complex cross section for calibration\n' \
                'There must be one dominating layup with a 90% contribution' \
                'to the reinforcement ration of the composite cross section')

        # Extract the characteristics of the dominating layout
        cm_key = self.concrete_mixture_key

        flu = self.fabric_layup_list[self.max_rho_idx]
        orientation_fn_key = flu.orientation_fn_key

        flo_key = flu.fabric_layout_key
        s_tex_z = flu.s_tex_z

        # construct the key of the composite cross section unit cell
        # and store it
        #
        key = cm_key + '_' + flo_key + '_' + \
            '%.5f' % (s_tex_z) + '_' + orientation_fn_key

        ccsuc = CCSUnitCell.db.get(key, None)
        if ccsuc is None:
            ccsuc = CCSUnitCell(concrete_mixture_key=cm_key,
                                fabric_layout_key=flo_key,
                                s_tex_z=s_tex_z,
                                orientation_fn_key=orientation_fn_key,
                                rho=self.rho_c)
            CCSUnitCell.db[key] = ccsuc

        print('CCSUnitCell: stored parameters for', key)
        ccsuc.set_param(material_model, calibration_test, df)
        CCSUnitCell.db[key].save()

    # -------------------------------------------------------------------------
    # view
    # -------------------------------------------------------------------------

    traits_view = View(Item('key', style='readonly', show_label=False),
                       VSplit(
        VGroup(
            Spring(),
            Item(
                'concrete_mixture_key', label='concrete mixture'),
            Spring(),
            Item('concrete_mixture_ref@', show_label=False),
            Spring(),
            id='exdb.ccs.cm',
            label='concrete mixture',
            dock='tab',
        ),
        HSplit(
            Item('fabric_layup_list@', editor=ccs_table_editor,
                 show_label=False, resizable=True),
            Item('selected_layup@',
                 show_label=False, resizable=True),
            # label = 'fabric layup',
            # id = 'exdb.ccs.flu',
            #                              scrollable = True,
            #                              dock = 'tab',
        ),
        Group(
            Item('thickness',
                 style='readonly', show_label=True, format_str="%.3f"),
            Item('rho_c',
                 style='readonly', show_label=True, format_str="%.4f"),
            Item('E_c28',
                 style='readonly', show_label=True, format_str="%.0f"),
            Item('is_regular',
                 style='readonly', show_label=True),
            label='derived params',
            id='exdb.ccs.dp',
            dock='tab',
        ),

        dock='tab',
        id='ccs.db.vsplit',
        orientation='vertical',
    ),
        dock='tab',
        id='ccs.db',
        scrollable=True,
        resizable=True,
        height=0.4,
        width=0.5,
        buttons=['OK', 'Cancel'],
    )


# convenience definition of a layup consisting of pure concrete
#
def plain_concrete(thickness):
    return FabricLayUp(
        n_layers=1,
        s_tex_z=thickness,
        fabric_layout_key='unreinforced',
        orientation_fn_key='unreinforced'
    )

# --------------------------------------------------------------------------

if __name__ == '__main__':

    s_tex_z = 0.030 / (7 + 1)
    ccs = CompositeCrossSection(
        fabric_layup_list=[
            plain_concrete(s_tex_z * 0.5),
            FabricLayUp(
                n_layers=7,
                orientation_fn_key='unreinforced',
                s_tex_z=s_tex_z,
                fabric_layout_key='MAG-07-03'
            ),
            plain_concrete(s_tex_z * 0.5)
        ],
        concrete_mixture_key='PZ-0708-1'
    )

    ccs.configure_traits()
