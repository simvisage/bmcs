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
# Created on Apr 15, 2010 by: rch

from enthought.traits.api import \
    HasTraits, Dict, Str, Enum, Instance, Int, Class, Type, \
    Directory, List, Property, Float, cached_property, \
    Event, on_trait_change

from enthought.traits.ui.api import \
    View, Item, Tabbed, VGroup, HGroup, ModelView, HSplit, VSplit, \
    CheckListEditor, EnumEditor, TableEditor, TabularEditor, Handler, \
    Group

from enthought.traits.ui.menu import \
    Action, CloseAction, HelpAction, Menu, \
    MenuBar, NoButtons, Separator, ToolBar

from enthought.traits.ui.tabular_adapter \
    import TabularAdapter

from enthought.traits.ui.table_filter \
    import EvalFilterTemplate, MenuFilterTemplate, RuleFilterTemplate, \
           EvalTableFilter, MenuTableFilter

from enthought.traits.ui.table_column import \
    ObjectColumn

from enthought.traits.ui.tabular_adapter \
    import TabularAdapter, AnITabularAdapter

from util.find_class import \
    _find_class

from util.traits.editors.mpl_figure_editor import \
    MPLFigureEditor

from matplotlib.figure import \
    Figure

from matplotlib.pyplot import \
     plot

from ex_run import ExRun

from ex_type import ExType

from promod.simdb import \
    SimDB

import os, fnmatch
import pickle
import string

# Access to the toplevel directory of the database
#
simdb = SimDB()

class ExRunTableAdapter ( TabularAdapter ):

    columns = Property

    #---------------------------------------------------------------
    # EXTRACT FACTOR NAMES AS COLUMNS FOR TABLE EDITOR
    #-------------------------------------------------------------------
    def _get_columns( self ):
        cols = [ ( 'index', 'index' ) ] # , ('key', 'key') ]
        obj = self.object
        for field_name in obj.field_names:
            cols.append( ( field_name, field_name ) )
        print cols
        return cols

    selection_mode = 'rows',

    font = 'Courier 10'
    alignment = 'right'
    odd_bg_color = 'lightblue'

    index_width = Float( 50 )

    index_text = Property
    def _get_index_text ( self ):
        return str( self.row )

    key_width = Float( 120 )
    key_text = Property
    def _get_key_text( self ):
        factor_idx = self.column - 1
        value = self.object.instances[ self.row, factor_idx ]
        return str( value )

exdb_tabular_editor = TabularEditor( adapter = ExRunTableAdapter(),
                                   multi_select = True,
                                   selected = 'selected_instances',
                                 )

exdb_table_editor = TableEditor( 
                        columns_name = 'table_columns',
                        selection_mode = 'rows',
                        selected = 'object.selected_instances',
                        # selected_indices  = 'object.selected_exruns',
                        show_toolbar = True,
                        auto_add = False,
                        configurable = True,
                        sortable = True,
                        reorderable = False,
                        sort_model = False,
                        orientation = 'vertical',
                        auto_size = True,
                        filters = [EvalFilterTemplate,
                                       MenuFilterTemplate,
                                       RuleFilterTemplate ],
                        search = EvalTableFilter(),
            )

#------------------------------------------------------------------------------------------
# Class Extension - global persistent container of class instances
#------------------------------------------------------------------------------------------
class ExRunClassExt( HasTraits ):

    category = Str( 'exdata' )

    path = List( [] )

    # dictionary of predefined instances - used for 
    # debugging and early stages of class developmemnt.
    #
    klass = Type

    classname = Property( depends_on = 'klass' )
    @cached_property
    def _get_classname( self ):
        return self.klass.__name__

    field_names = Property( depends_on = 'klass' )
    @cached_property
    def _get_field_names( self ):
        '''
        Get the list of table fields.
        '''
        return self.klass.class_trait_names( table_field = lambda x: x != None )

    # Get columns for the table editor
    #
    table_columns = Property( depends_on = 'klass' )
    @cached_property
    def _get_table_columns( self ):
        columns = [ ObjectColumn( name = 'key',
                                  editable = False,
                                  horizontal_alignment = 'center',
                               )  ]
        columns += [ ObjectColumn( name = field_name,
                               editable = False,
                               horizontal_alignment = 'center',
                               #width = 100 
                               ) for field_name in self.field_names ]
        return columns

    dir = Directory
    def _dir_default( self ):
        '''Name of the directory for the data of the class
        '''
        full_path = ( simdb.simdb_dir, self.category )
        path = os.path.join( *full_path )
        return path

    def _get_file_list( self ):
        '''Populate the instances with the values
        '''
        # walk through the directories and read the 
        # values
        #
        for root, sub_folders, files in os.walk( self.dir ):
            for folder in sub_folders:
                ex_type_file = os.path.join( root, folder, 'ex_type.cls' )
                if os.path.exists( ex_type_file ):

                    # check if the class specification coincides with
                    # the klass trait of this table
                    #
                    f = open( ex_type_file, 'r' )
                    ex_type_klass = f.read().split( '\n' )[0] # use trim here
                    f.close()

                    klass = _find_class( ex_type_klass )
                    if klass == self.klass:

                        file_ext = klass.file_ext
                        ex_type_dir = os.path.join( root, folder )
                        ex_type_files = os.listdir( ex_type_dir )
                        for filename in fnmatch.filter( ex_type_files, '*.%s' % file_ext ):
                            yield os.path.join( ex_type_dir, filename )

    ex_run_list = List
    def _ex_run_list_default( self ):
        return [ ExRun( ex_run_file ) for ex_run_file in self._get_file_list() ]

    instances = List
    def _instances_default( self ):
        return [ ex_run.ex_type for ex_run in self.ex_run_list ]

    selected_instances = List
    def _selected_instances_default( self ):
        return []

    selected_instance = Property( Instance( ExType ),
                                  depends_on = 'selected_instances[]' )
    def _get_selected_instance( self ):
        if len( self.selected_instances ) > 0:
            return self.selected_instances[0]
        else:
            return None

    def keys( self ):
        return self.instances.keys()

    def get( self, name, Missing ):
        it = self.instances.get( name, Missing )
        return it

    #-------------------------------------------------------------------
    # PLOT OBJECT
    #-------------------------------------------------------------------

    figure = Instance( Figure )
    def _figure_default( self ):
        figure = Figure( facecolor = 'white' )
        #figure.add_axes( [0.08, 0.13, 0.85, 0.74] )
        figure.add_axes( [0.15, 0.15, 0.75, 0.75] )
        return figure

    # event to trigger the replotting - used by the figure editor
    # 
    data_changed = Event

    # selected plot template
    #
    plot_template = Enum( values = 'plot_template_list' )

    # list of availble plot templates 
    # (gets extracted from the model whenever it's been changed)
    #
    plot_template_list = Property( depends_on = 'klass' )
    @cached_property
    def _get_plot_template_list( self ):
        '''Change the selection list of plot templates.
        
        This method is called upon every change of the model. This makes the viewing of
        different experiment types possible.
        '''
        return self.instances[0].plot_templates.keys()

    @on_trait_change( 'selected_instances,plot_template' )
    def redraw( self, ui_info = None ):
        ''' Use the currently selected plot template to plot it in the Figure.
        '''

        # map the array dimensions to the plot axes
        #
        figure = self.figure

        axes = figure.gca()
        axes.clear()

        # get the labels (keys) of the selected instances for 
        # attibution of the legends in matplotlib:
        #

        for run in self.selected_instances:
            proc_name = run.plot_templates[ self.plot_template ]
            plot_processor = getattr( run, proc_name )
            plot_processor( axes )

        legend_names = []
        for run in self.selected_instances:
            legend_names = legend_names + [run.key]
        axes.legend( legend_names, loc = 7 )


        self.data_changed = True

    #---------------------------------------------------------------------------------
    # VIEW
    #---------------------------------------------------------------------------------

    traits_view = View( 
                    HSplit( 
                       VSplit( 
                               VGroup( 
                                     HGroup( Item( 'classname',
                                                 emphasized = True,
                                                 style = 'readonly',
                                                 label = 'database extension class' )
                                            ),
                                     Item( 'instances',
                                           editor = exdb_table_editor,
                                           show_label = False,
                                           style = 'custom' ,
                                           resizable = True ),
                                     ),
                                label = 'experiment table',
                                id = 'exdb.table.instances',
                                dock = 'tab',
                               scrollable = True,
                             ),
                       VGroup( 
                           VGroup( 
                                 Item( 'selected_instance@',
                                       resizable = True,
                                       show_label = False ),
                                 label = 'experiment',
                                 id = 'exdb.table.instance',
                                 dock = 'tab',
                                 scrollable = True,
                                 ),
                            Group( 
                                Item( 'figure', editor = MPLFigureEditor(),
                                     resizable = True, show_label = False ),
                                     id = 'exrun_table.plot_sheet',
                                     label = 'plot sheet',
                                     dock = 'tab',
                                     ),
                            Group( 
                                Item( 'plot_template' ),
                                columns = 1,
                                label = 'plot parameters',
                                id = 'exrun_table.plot_params',
                                dock = 'tab',
                                ),
                            id = 'exdb.table.right',
                            layout = 'split',
                            label = 'selected instance',
                            dock = 'tab',
                            ),
                        id = 'exdb.table.splitter',
                    ),
                    id = 'exdb.table',
                    buttons = ['OK', 'Cancel'],
                    dock = 'tab',
                    resizable = True,
                    height = 0.8, width = 0.8,
                    )

if __name__ == '__main__':
    from promod.exdb.ex_composite_tensile_test import \
        ExCompositeTensileTest
    from promod.exdb.ex_plate_test import \
        ExPlateTest
    from promod.exdb.ex_bending_test import \
        ExBendingTest
    ex = ExRunClassExt( klass = ExCompositeTensileTest )
#    ex = ExRunClassExt( klass = ExPlateTest )
#    ex = ExRunClassExt( klass = ExBendingTest )
    ex.configure_traits()
