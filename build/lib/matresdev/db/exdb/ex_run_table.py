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
# Created on Apr 15, 2010 by: rch

import fnmatch
from matplotlib.figure import \
    Figure
import os

from traits.api import \
    Str, Enum, Instance, Type, \
    Directory, List, Property, Float, cached_property, \
    Event, on_trait_change
from traitsui.api import \
    View, Item, VGroup, HGroup, HSplit, VSplit, \
    TableEditor, TabularEditor,  \
    Group
from traitsui.table_column import \
    ObjectColumn
from traitsui.table_filter \
    import EvalFilterTemplate, MenuFilterTemplate, RuleFilterTemplate, \
    EvalTableFilter
from traitsui.tabular_adapter \
    import TabularAdapter

from .ex_run import ExRun
from .ex_type import ExType
from matresdev.db.simdb import \
    SimDBClassExt
from matresdev.db.simdb.simdb import \
    simdb
from util.find_class import \
    _find_class
from util.traits.editors.mpl_figure_editor import \
    MPLFigureEditor


class ExRunTableAdapter (TabularAdapter):

    columns = Property

    #---------------------------------------------------------------
    # EXTRACT FACTOR NAMES AS COLUMNS FOR TABLE EDITOR
    #-------------------------------------------------------------------
    def _get_columns(self):
        cols = [('index', 'index')]  # , ('key', 'key') ]
        obj = self.object
        for field_name in obj.field_names:
            cols.append((field_name, field_name))
        print(cols)
        return cols

    selection_mode = 'rows',

    font = 'Courier 10'
    alignment = 'right'
    odd_bg_color = 'lightblue'

    index_width = Float(50)

    index_text = Property

    def _get_index_text(self):
        return str(self.row)

    key_width = Float(120)
    key_text = Property

    def _get_key_text(self):
        factor_idx = self.column - 1
        value = self.object.inst_list[self.row, factor_idx]
        return str(value)

exdb_tabular_editor = TabularEditor(adapter=ExRunTableAdapter(),
                                    multi_select=True,
                                    selected='selected_inst_list',
                                    )

exdb_table_editor = TableEditor(
    columns_name='table_columns',
    selection_mode='rows',
    selected='object.selected_inst_list',
    # selected_indices  = 'object.selected_exruns',
    show_toolbar=True,
    auto_add=False,
    configurable=True,
    sortable=True,
    reorderable=False,
    sort_model=False,
    orientation='vertical',
    auto_size=True,
    filters=[EvalFilterTemplate,
             MenuFilterTemplate,
             RuleFilterTemplate],
    search=EvalTableFilter(),
)

#-------------------------------------------------------------------------
# Class Extension - global persistent container of class inst_list
#-------------------------------------------------------------------------


class ExRunClassExt(SimDBClassExt):

    category = Str('exdata')

    path = List([])

    # dictionary of predefined inst_list - used for
    # debugging and early stages of class developmemnt.
    #
    klass = Type

    classname = Property(depends_on='klass')

    @cached_property
    def _get_classname(self):
        return self.klass.__name__

    field_names = Property(depends_on='klass')

    @cached_property
    def _get_field_names(self):
        '''
        Get the list of table fields.
        '''
        return self.klass.class_trait_names(table_field=lambda x: x != None)

    # Get columns for the table editor
    #
    table_columns = Property(depends_on='klass')

    @cached_property
    def _get_table_columns(self):
        columns = [ObjectColumn(name='key',
                                editable=False,
                                horizontal_alignment='center',
                                )]
        columns += [ObjectColumn(name=field_name,
                                 editable=False,
                                 horizontal_alignment='center',
                                 # width = 100
                                 ) for field_name in self.field_names]
        return columns

    dir = Directory

    def _dir_default(self):
        '''Name of the directory for the data of the class
        '''
        full_path = (simdb.simdb_dir, self.category)
        path = os.path.join(*full_path)
        return path

    def _get_file_list(self):
        '''Populate the inst_list with the values
        '''
        # walk through the directories and read the
        # values
        #
        print('file list', self.dir)
        for root, sub_folders, files in os.walk(self.dir):
            for folder in sub_folders:
                ex_type_file = os.path.join(root, folder, 'ex_type.cls')
                if os.path.exists(ex_type_file):

                    # check if the class specification coincides with
                    # the klass trait of this table
                    #
                    f = open(ex_type_file, 'r')
                    ex_type_klass = f.read().split('\n')[0]  # use trim here
                    f.close()

                    klass = _find_class(ex_type_klass)

                    if klass and klass.__name__ == self.klass.__name__:

                        file_ext = klass().file_ext
                        ex_type_dir = os.path.join(root, folder)
                        ex_type_files = os.listdir(ex_type_dir)
                        for filename in fnmatch.filter(ex_type_files, '*.%s' % file_ext):
                            yield os.path.join(ex_type_dir, filename)

    ex_run_list = List

    def _ex_run_list_default(self):
        print('file_list', self._get_file_list())
        return [ExRun(ex_run_file) for ex_run_file in self._get_file_list()]

    inst_list = List

    def _inst_list_default(self):
        print('ex_run_list', self.ex_run_list)
        return [ex_run.ex_type for ex_run in self.ex_run_list]

    selected_inst_list = List

    def _selected_inst_list_default(self):
        return []

    selected_instance = Property(Instance(ExType),
                                 depends_on='selected_inst_list[]')

    def _get_selected_instance(self):
        if len(self.selected_inst_list) > 0:
            return self.selected_inst_list[0]
        else:
            return None

    def export_inst_list(self):
        ex_table = []
        for inst in self.inst_list:
            row = [getattr(inst, tcol) for tcol in self.table_columns]
            ex_table.append(row)
        print(ex_table)

    def keys(self):
        return list(self.inst_list.keys())

    def get(self, name, Missing):
        it = self.inst_list.get(name, Missing)
        return it

    #-------------------------------------------------------------------
    # PLOT OBJECT
    #-------------------------------------------------------------------

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        # figure.add_axes( [0.08, 0.13, 0.85, 0.74] )
        figure.add_axes([0.15, 0.15, 0.75, 0.75])
        return figure

    # event to trigger the replotting - used by the figure editor
    #
    data_changed = Event

    # selected plot template
    #
    plot_template = Enum(values='plot_template_list')

    # list of availble plot templates
    # (gets extracted from the model whenever it's been changed)
    #
    plot_template_list = Property(depends_on='klass')

    @cached_property
    def _get_plot_template_list(self):
        '''Change the selection list of plot templates.

        This method is called upon every change of the model. This makes the viewing of
        different experiment types possible.
        '''
        return list(self.inst_list[0].plot_templates.keys())

    @on_trait_change('selected_inst_list,plot_template')
    def redraw(self, ui_info=None):
        ''' Use the currently selected plot template to plot it in the Figure.
        '''

        # map the array dimensions to the plot axes
        #
        figure = self.figure

        axes = figure.gca()
        axes.clear()

        # get the labels (keys) of the selected inst_list for
        # attibution of the legends in matplotlib:
        #

        for run in self.selected_inst_list:
            proc_name = run.plot_templates[self.plot_template]
            plot_processor = getattr(run, proc_name)
            plot_processor(axes)

        legend_names = []
        for run in self.selected_inst_list:
            legend_names = legend_names + [run.key]
        if len(legend_names) > 0:
            axes.legend(legend_names, loc=7)

        self.data_changed = True

    #-------------------------------------------------------------------------
    # VIEW
    #-------------------------------------------------------------------------

    traits_view = View(
        HSplit(
            VSplit(
                VGroup(
                    HGroup(Item('classname',
                                emphasized=True,
                                style='readonly',
                                label='database extension class')
                           ),
                    Item('inst_list',
                         editor=exdb_table_editor,
                         show_label=False,
                         style='custom',
                         resizable=True),
                ),
                label='experiment table',
                id='exdb.table.inst_list',
                dock='tab',
                scrollable=True,
            ),
            VGroup(
                VGroup(
                    Item('selected_instance@',
                         resizable=True,
                         show_label=False),
                    label='experiment',
                    id='exdb.table.instance',
                    dock='tab',
                    scrollable=True,
                ),
                Group(
                    Item('figure', editor=MPLFigureEditor(),
                         resizable=True, show_label=False),
                    id='exrun_table.plot_sheet',
                    label='plot sheet',
                    dock='tab',
                ),
                Group(
                    Item('plot_template'),
                    columns=1,
                    label='plot parameters',
                    id='exrun_table.plot_params',
                    dock='tab',
                ),
                id='exdb.table.right',
                layout='split',
                label='selected instance',
                dock='tab',
            ),
            id='exdb.table.splitter',
        ),
        id='exdb.table',
        buttons=['OK', 'Cancel'],
        dock='tab',
        resizable=True,
        height=0.8, width=0.8,
    )
