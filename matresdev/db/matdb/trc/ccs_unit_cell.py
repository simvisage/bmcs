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
# Created on May 12, 2010 by: rch

from traits.api import \
    HasTraits, List, Float, \
    on_trait_change, Instance, \
    Str, Property, cached_property, \
    Event, \
    DelegatesTo, Button

from traitsui.api import \
    View, Item, DirectoryEditor, TabularEditor, HSplit, VGroup, \
    TableEditor, EnumEditor, Handler, FileEditor, VSplit, Group, \
    InstanceEditor, HGroup, Spring

from traitsui.tabular_adapter \
    import TabularAdapter

from traitsui.table_column import \
    ObjectColumn

# overload the 'get_label' method from 'Item' to display units in the label
from util.traits.ui.item import \
    Item

from mathkit.mfn.mfn_line.mfn_line import \
    MFnLineArray

from matplotlib.figure import \
    Figure

from util.traits.editors.mpl_figure_editor import \
    MPLFigureEditor

from numpy import \
    array, ones_like, frompyfunc, linspace

import numpy as np

from .concrete_mixture \
    import ConcreteMixture

from .fabric_layout \
    import FabricLayOut

from .fabric_layup \
    import FabricLayUp

from matresdev.db.simdb.simdb_class import \
    SimDBClass, SimDBClassExt

class DamageFunctionEntry(HasTraits):
    material_model = Str(input=True)
    calibration_test = Str(input=True)
    damage_function = Instance(MFnLineArray, input=True)

    input_change = Event
    @on_trait_change('+input')
    def _set_input_change(self):
        self.input_change = True


df_table_editor = TableEditor(
                        columns=[ ObjectColumn(name='material_model',
                                                        editable=False,
                                                        horizontal_alignment='center'),
                                          ObjectColumn(name='calibration_test',
                                                        editable=False,
                                                        horizontal_alignment='center') ],
                        selection_mode='rows',
                        selected='object.selected_dfs',
                        deletable=True,
                        editable=False,
                        show_toolbar=True,
                        auto_add=False,
                        configurable=False,
                        sortable=False,
                        reorderable=False,
                        sort_model=False,
                        orientation='vertical',
                        auto_size=True,
            )

ccsuc_table_editor = TableEditor(
                        columns=[ ObjectColumn(name='fabric_layout_key',
                                                  editable=False, width=0.20),
                                    ObjectColumn(name='concrete_mixture_key',
                                                  editable=False, width=0.20),
                                    ObjectColumn(name='s_tex_z',
                                                  editable=False, width=0.20),
                                    ],
                        selection_mode='row',
                        selected='object.selected_ccuc',
                        show_toolbar=True,
                        auto_add=False,
                        configurable=True,
                        sortable=True,
                        deletable=False,
                        reorderable=True,
                        sort_model=False,
                        auto_size=False,
#                        filters      = [EvalFilterTemplate,
#                                       MenuFilterTemplate,
#                                       RuleFilterTemplate ],
#                        search       = EvalTableFilter(),
            )

class CCSUnitCell(SimDBClass):
    '''Describes the combination of the concrete mixture and textile cross section.
    Note: only one concrete can be defined for the entire composite cross section.
    '''

    input_change = Event
    @on_trait_change('+input, damage_function_list.input_change')
    def _set_input_change(self):
        self.input_change = True

    #--------------------------------------------------------------------------------
    # select the concrete mixture from the concrete database:
    #--------------------------------------------------------------------------------

    concrete_mixture_key = Str(simdb=True, input=True, table_field=True,
                                auto_set=False, enter_set=True)

    concrete_mixture_ref = Property(Instance(SimDBClass),
                                     depends_on='concrete_mixture_key')
    @cached_property
    def _get_concrete_mixture_ref(self):
        return ConcreteMixture.db[ self.concrete_mixture_key ]

    #--------------------------------------------------------------------
    # textile fabric layout used
    #--------------------------------------------------------------------
    #
    fabric_layout_key = Str(simdb=True, input=True, table_field=True,
                             auto_set=False, enter_set=True)
    def _fabric_layout_key_default(self):
        return list(FabricLayOut.db.keys())[0]

    fabric_layout_ref = Property(Instance(SimDBClass),
                                  depends_on='fabric_layout_key')
    @cached_property
    def _get_fabric_layout_ref(self):
        return FabricLayOut.db[ self.fabric_layout_key ]

    # equidistant spacing in thickness direction (z-direction) between the layers
    #
    s_tex_z = Float(1, unit='m', simdb=True, input=True, table_field=True,
                      auto_set=False, enter_set=True)

    # orientation function of the layup fabric
    #
    orientation_fn_key = Str('', simdb=True, table_field=True,
                                auto_set=False, enter_set=True)

    #--------------------------------------------------------------------------------
    # Derived material parameters - supplied during the calibration
    #--------------------------------------------------------------------------------

    def get_E_m_time(self, age):
        '''function for the composite E-modulus as weighted sum of all
        fabric layups. Returns a function depending of the concrete age.
        '''
        E_m = self.concrete_mixture_ref.get_E_m_time(age)
        print('E_m_time', E_m)
        return E_m

    def get_E_c_time(self, age):
        '''function for the composite E-modulus as weighted sum of all
        fabric layups. Returns a function depending of the concrete age.
        '''
        rho = self.rho
        E_tex = self.E_tex
        E_m = self.concrete_mixture_ref.get_E_m_time(age)
        E_c = (1 - rho) * E_m + rho * E_tex
        return E_c

    E_c28 = Property(Float, unit='MPa', depends_on='input_change')
    @cached_property
    def _get_E_c28(self):
        '''Composite E-modulus after 28 days.
        '''
        return self.get_E_c_time(28)

    nu = DelegatesTo('concrete_mixture_ref', listenable=False)

    #--------------------------------------------------------------------------------
    # Defined material parameters - supplied during the calibration
    #--------------------------------------------------------------------------------
    rho = Float

    E_tex = Property(depends_on='input_changed')
    def _get_E_tex(self):
        return self.fabric_layout_ref.E_tex_0

    #--------------------------------------------------------------------------------
    # Bag for material parameters
    #--------------------------------------------------------------------------------
    #
    # The material parameters are quantities to be used by models that should make
    # prognosis of the behavior in general condition. The composite cross section
    # can be modeled by a variety of models, each of which may need different
    # parameters. The parameters are determined using some calibration test.
    # Therefore, in order to trace back the origin of the material parameters
    # it is necessary to store the value of the material parameter together with
    # the model that has calibrated it and by the test setup that was used
    # for calibration. In this way, the range of validity of the calibrated damage
    # function is available.
    #
    # The material parameters are stored in a list with the combined key
    # containing the tuple (material model, test run).
    #
    damage_function_list = List(DamageFunctionEntry, input=True)

    def set_param(self, material_model, calibration_test, df):
        print('adding damage function')
        for mp_entry in self.damage_function_list:
            if (mp_entry.material_model == material_model and
                 mp_entry.calibration_test == calibration_test):
                mp_entry.damage_function = df
                print('only the function updated')
                return
        mp_entry = DamageFunctionEntry(material_model=material_model,
                                        calibration_test=calibration_test,
                                        damage_function=df)
        self.damage_function_list.append(mp_entry)
        print('new function added')

    def get_param(self, material_model, calibration_test):
        print('material_model', material_model)
        print('calibration_test', calibration_test)
        for mp_entry in self.damage_function_list:
            if (mp_entry.material_model == material_model and
                 mp_entry.calibration_test == calibration_test):
                return mp_entry.damage_function

        raise ValueError('no entry in unit cell with key ( %s ) for model ( %s ) and test( %s )' % (self.key, material_model, calibration_test))

    #-------------------------------------------------------------
    # derive the average phi-function based on all entries
    # in damage_function_list
    #-------------------------------------------------------------

    df_average = Property(Instance(MFnLineArray),
                           depends_on='damage_function_list')
    @cached_property
    def get_df_average(self, n_points):
        '''derive the average phi-function based on all entries
        in damage_function_list
        '''

        def get_y_average(self, x_average):
            '''get the y-values from the mfn-functions in df_list for
            'x_average' and return the average.
            Note that the shape of 'mfn.xdata' does not necessarily needs to be equal in all
            'DamageFunctionEntries' as the number of steps used for calibration or the adaptive
            refinement in 'tloop' might have been different for each case.
            '''
            y_list = [ self.damage_function_list[i].damage_function.get_value(x_average) \
                       for i in range(len(self.damage_function_list)) ]
            return sum(y_list) / len(y_list)

        get_y_average_vectorized = frompyfunc(get_y_average, 2, 1)

        mfn = MFnLineArray()

        # take the smallest value of the strains for the average function. Beyond this value
        # the average does not make sense anymore because it depends on the arbitrary number
        # of entries in the df_list
        #
        xdata_min = min(self.damage_function_list[i].damage_function.xdata[-1] \
                         for i in range(len(self.damage_function_list)))

        # number of sampling point used for the average phi function
        #
        mfn.xdata = linspace(0., xdata_min, num=n_points)

        # get the corresponding average ydata values
        #
        mfn.ydata = self.get_y_average_vectorized(mfn.xdata)

        return mfn


    #-------------------------------------------------------------------


    #-------------------------------------------------------------------
    # PLOT OBJECT
    #-------------------------------------------------------------------

    figure = Instance(Figure, transient=True)
    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.add_axes([0.08, 0.13, 0.85, 0.74])
        return figure

    selected_dfs = List(Instance(DamageFunctionEntry), transient=True)
    def _selected_dfs_default(self):
        return self.damage_function_list

    print_values = Button()
    def _print_values_fired(self):
        for mp_entry in self.selected_dfs:
            print('x', mp_entry.damage_function.xdata)
            print('y', mp_entry.damage_function.ydata)

    save_phi_fn_values = Button()
    def _save_phi_fn_values_fired(self):
        'save phi_fn xy-data to file'
        print('save phi_arr data to file')
        for mp_entry in self.selected_dfs:
            xdata = mp_entry.damage_function.xdata
            ydata = mp_entry.damage_function.ydata
            phi_arr = np.hstack([xdata[:, None], ydata[:, None]])
            print('phi_arr ', phi_arr.shape, ' save to file "phi_arr.csv"')
            np.savetxt('phi_arr.csv', phi_arr, delimiter=';')

    # event to trigger the replotting - used by the figure editor
    #
    data_changed = Event

#    @on_trait_change('damage_function_list')
    @on_trait_change('input_change, selected_dfs')
    def _redraw(self):
        # map the array dimensions to the plot axes
        #
        figure = self.figure
        axes = figure.gca()
        axes.clear()

        for mp_entry in self.selected_dfs:
#        for mp_entry in self.damage_function_list:
            df = mp_entry.damage_function
            mp_entry.damage_function.mpl_plot(axes)

        self.data_changed = True

    #--------------------------------------------------------------------------------
    # view
    #--------------------------------------------------------------------------------

    traits_view = View(Item('key', style='readonly'),
                       Item('s_tex_z', style='readonly', format_str="%.5f"),
                        Item('orientation_fn_key', style='readonly'),
                        Item('rho', style='readonly', format_str="%.5f"),
                        VSplit(
                        VGroup(
                            Spring(),
                            Item('concrete_mixture_key', style='readonly',
                                label='concrete mixture'),
                            Spring(),
                            Item('concrete_mixture_ref@', style='readonly',
                                 show_label=False),
                            Spring(),
                            id='exdb.ccsuc.cm',
                            label='concrete mixture',
                            dock='tab',
                            ),
                        VGroup(
                            Spring(),
                            Item('fabric_layout_key', style='readonly',
                                label='fabric layout'),
                            Spring(),
                            Item('fabric_layout_ref@', style='readonly',
                                show_label=False),
                            Spring(),
                            id='exdb.ccsuc.lo',
                            label='fabric layout',
                            dock='tab',
                            ),
                        Group(
                              Item('print_values', show_label=False),
                              Item('save_phi_fn_values', show_label=False),
                        Item('damage_function_list@', editor=df_table_editor,
                             resizable=True, show_label=False),
                             id='exdb.ccsuc.damage_functions',
                             label='damage functions',
                             dock='tab',
                        ),
                        Group(
                        Item('figure', editor=MPLFigureEditor(),
                             resizable=True, show_label=False),
                             id='exdb.ccsuc.plot_sheet',
                             label='plot sheet',
                             dock='tab',

                        ),
#                        Group(
#                        Item('rho_c',
#                             style = 'readonly', show_label = True, format_str="%.4f"),
#                        Item('E_c28',
#                             style = 'readonly', show_label = True, format_str="%.0f" ),
#                        label = 'derived params',
#                        id = 'exdb.ccsuc.dp',
#                        dock = 'tab',
#                        ),

                        dock='tab',
                        id='exdb.ccsuc.db.vsplit',
                        orientation='vertical',
                        ),
                        dock='tab',
                        id='ccsuc.db',
                        scrollable=True,
                        resizable=True,
                        height=0.4,
                        width=0.5,
                        buttons=['OK', 'Cancel'],
                        )

# Setup the database class extension
#
CCSUnitCell.db = SimDBClassExt(
            klass=CCSUnitCell,
            verbose='io',
            )

if __name__ == '__main__':

    CCSUnitCell.db.configure_traits()
