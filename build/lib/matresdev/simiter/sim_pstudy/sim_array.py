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
# Created on Jan 19, 2010 by: rch

# @todo:
# - test case for levels_modified and for model_modified
# - other_factors should be explicitly listed - concept for level status
#   change
# - legend for depth axis in the plots - done
# - show the output values in the input_table
#   in terms of DOE the input_table corresponds to the full_factorial design.
# - window menu (exit menu entry in the File menu)
# - threaded evaluation (estimation of computation time of a single run)
# - progress bar showing the number of simulations
# - purging of the factor space using regression / splines?
#   this would require the fractional factorial array
#
# Q: how to make a toolbar within a view without having to write a special editor?
# consider the tree on the left showing the studied space
#
import pickle

from numpy import \
    hstack, mgrid, c_, \
    zeros, arange
from pyface.api import ProgressDialog
from traits.api import \
    HasTraits, Float, Property, cached_property, \
    Instance, List, on_trait_change, Int, \
    Event, Str, Dict, Array, \
    implements
from traitsui.api import \
    View, Item, VGroup, HGroup, HSplit, VSplit, \
    EnumEditor, TabularEditor
from traitsui.tabular_adapter \
    import TabularAdapter

from .i_sim_array import ISimArray
from .i_sim_model import ISimModel
from .sim_factor import \
    SimFactor, SimFloatFactor, SimIntFactor, SimEnumFactor
from .sim_output import SimOut
from functools import reduce


class RunTableAdapter (TabularAdapter):

    columns = Property

    #---------------------------------------------------------------
    # EXTRACT FACTOR NAMES AS COLUMNS FOR TABLE EDITOR
    #-------------------------------------------------------------------
    def _get_columns(self):
        cols = [('run', 'index')]
        ps = self.object
        if ps == None:
            return cols
        for factor_name, factor in zip(ps.factor_names, ps.factor_list):
            if isinstance(factor, SimEnumFactor):
                cols.append((factor_name, 'enum'))
            elif isinstance(factor, SimIntFactor):
                cols.append((factor_name, 'int'))
            elif isinstance(factor, SimFloatFactor):
                cols.append((factor_name, 'float'))
        for output_name in ps.output_names:
            cols.append((output_name, 'output'))

        return cols

    font = 'Courier 10'
    alignment = 'right'
    odd_bg_color = 'lightblue'

    index_width = Float(50)
    index_image = Property

    def _get_index_image(self):
        if self.object.output_table[self.row, 0]:
            return '@icons:red_ball'
        else:
            return None

    index_text = Property

    def _get_index_text(self):
        return str(self.row)

    float_width = Float(100)
    float_text = Property

    def _get_float_text(self):
        factor_idx = self.column - 1
        value = self.object.input_table[self.row, factor_idx]
        return '%6.2f' % value

    int_width = Float(60)
    int_text = Property

    def _get_int_text(self):
        factor_idx = self.column - 1
        value = self.object.input_table[self.row, factor_idx]
        return '%d' % value

    enum_width = Float(120)
    enum_text = Property

    def _get_enum_text(self):
        factor_idx = self.column - 1
        value = self.object.input_table[self.row, factor_idx]
        return str(value)

    output_width = Float(120)
    output_text = Property

    def _get_output_text(self):
        col_idx = self.column - 1
        value = self.object.run_table[self.row, col_idx]
        return str(value)

input_table_editor = TabularEditor(adapter=RunTableAdapter())


class SimArray(HasTraits):
    '''
    Parametric study in a model design space.

    This is a view to a parametric study on a supplied model.

    The model can be an arbitrary class that specifies a 
    * set of input factors and default ranges to be explored
    * set of output specifiers
    * peval function that returns an array of specified output values.

    The factors identified in the model are used to construct
    the factor ranges. By default, each factor is set to inactive 
    so that the parametric study outputs in a single-value level
    for each factor.

    The user can view any of the factors and specify if it should 
    be varied and also provide the levels.    
    '''
    implements(ISimArray)

    sim_model = Instance(ISimModel)

    def _sim_model_default(self):
        from .sim_model import SimModel
        return SimModel()

    #---------------------------------------------------------------
    # FACTOR LIST SPECIFICATION
    #-------------------------------------------------------------------
    factor_dict = Dict

    def _factor_dict_default(self):
        return self._get_factor_dict()

    def _sim_model_changed(self):
        self.factor_dict = self._get_factor_dict()

    def _get_factor_dict(self):
        '''
        Get the dictionary of factors provided by the simulation model.

        The factors are identified by the factor_levels metadata in the trait
        definition. For example

        my_param = Float( 20, factor_levels = (0, 10, 6) )

        specifies a float factor with  the levels [0,2,4,6,8,10]
        '''
        traits = self.sim_model.class_traits(ps_levels=lambda x: x != None)
        factor_dict = {}
        for tname, tval in list(traits.items()):
            if tval.is_trait_type(Int):
                min_l, max_l, n_l = tval.ps_levels
                pt = SimIntFactor(
                    min_level=min_l, max_level=max_l, n_levels=n_l)
            elif tval.is_trait_type(Float):
                min_l, max_l, n_l = tval.ps_levels
                pt = SimFloatFactor(
                    min_level=min_l, max_level=max_l, n_levels=n_l)
            else:
                pt = SimEnumFactor(model=self.sim_model,
                                   levels=tval.ps_levels)

            factor_dict[tname] = pt
        return factor_dict

    n_factors = Property

    def _get_n_factors(self):
        return len(self.factor_dict)

    # alphabetically ordered names of factor ranges
    #
    factor_names = Property(depends_on='factor_dict')

    @cached_property
    def _get_factor_names(self):
        names = list(self.factor_dict.keys())
        names.sort()
        return names

    # alphabetically ordered list of factor ranges
    #
    factor_list = Property(depends_on='factor_dict')

    @cached_property
    def _get_factor_list(self):
        return [self.factor_dict[name] for name in self.factor_names]

    #---------------------------------------------------------------
    # SELECTED FACTOR FOR EDITING
    #-------------------------------------------------------------------
    # Edit factor levels to be associated inn the study
    #
    selected_factor_name = Str(factor_modified=True, transient=True)

    def _selected_factor_name_default(self):
        return self.factor_names[0]

    def _selected_factor_name_changed(self):
        self.selected_factor = self.factor_dict[self.selected_factor_name]
    selected_factor = Instance(SimFactor, transient=True)

    def _selected_factor_default(self):
        return self.factor_dict[self.factor_names[0]]

    #---------------------------------------------------------------
    # FULL FACTORIAL SPECIFICATION
    #-------------------------------------------------------------------
    #
    levels2run = Property(Array('int_'))

    def _get_levels2run(self):
        '''Get the mapping between index of the factor level
           and the index of the run.

        levels2run[ factor1_level_idx, factor2_level_idx, ... ] = run_idx 

        '''
        # get number of levels for each factor
        #
        n_levels_list = [factor.get_n_levels() for factor in self.factor_list]
        size = reduce(lambda x, y: x * y, n_levels_list)
        levels_sizes = tuple([n_levels for n_levels in n_levels_list])
        levels2run = arange(size, dtype='int_')
        levels2run = levels2run.reshape(levels_sizes)
        return levels2run

    run2levels = Property(Array('int_'))

    def _get_run2levels(self):
        '''Get the table of runs with rows containing the level indexes 
        within the factor levels list.

        run2levels[ run_idx ] = [ factor1_level_idx, factor2_level_idx, ... ] 
        '''
        n_levels_list = [factor.get_n_levels() for factor in self.factor_list]
        levels_slices = tuple([slice(0, n_levels, 1)
                               for n_levels in n_levels_list])
        levels_grid = mgrid[levels_slices]

        run_idx_arr = c_[tuple([x.flatten() for x in levels_grid])]
        return run_idx_arr

    changed = Event

    @on_trait_change('factor_dict.+levels_modified,_output_cache')
    def _set_changed(self):
        print('new values calculated')
        self.changed = True

    # Get the permutation of all factor levels as an array
    #
    input_table = Property(Array, depends_on='factor_dict.+levels_modified')

    @cached_property
    def _get_input_table(self):
        '''Get the array containing the level values for each run.
        '''
        run_idx_arr = self._get_run2levels()
        run_levels_arr = zeros(run_idx_arr.shape, dtype=object)

        # construct the mapping between the indices of a factor
        # and level and the level value
        level_map = [[level
                      for level in factor.level_list]
                     for factor in self.factor_list]

        for run_idx, run in enumerate(run_idx_arr):
            for factor_idx, level_idx in enumerate(run):
                run_levels_arr[run_idx, factor_idx] = level_map[
                    factor_idx][level_idx]

        return run_levels_arr

    #---------------------------------------------------------------
    # RESULT ARRAY SPECIFICATION
    #-------------------------------------------------------------------
    outputs = Property(List(SimOut), depends_on='sim_model')

    @cached_property
    def _get_outputs(self):
        if self.sim_model == None:
            return []
        else:
            return self.sim_model.get_sim_outputs()

    # extract the available names
    output_names = Property(List(Str))

    def _get_output_names(self):
        return [r.name for r in self.outputs]

    # number of outputs
    n_outputs = Property(Int)

    def _get_n_outputs(self):
        return len(self.output_names)

    #-------------------------------------------------------------------------
    # SELECTED RESULT FOR EDITTING
    #-------------------------------------------------------------------------
    # active selection
    #
    selected_output_name = Str(transient=True)

    def _selected_output_name_default(self):
        return self.output_names[0]
    selected_output_idx = Property(Int, depends_on='selected_output_name')

    def _get_selected_output_idx(self):
        return self.output_names.index(self.selected_output_name)

    def _selected_output_name_changed(self):
        self.selected_output = self.outputs[self.selected_output_idx]

    selected_output = Instance(SimOut, transient=True)

    def _selected_output_default(self):
        return self.outputs[0]

    #-------------------------------------------------------------------------
    # OUTPUT ARRAY
    #-------------------------------------------------------------------------
    output_array = Property(
        Array, depends_on='factor_dict.+levels_modified,_output_cache')

    @cached_property
    def _get_output_array(self):
        '''Setup an array to accommodate the calculated values.

        The shape of the array respects the parametric space.
        Each dimension corresponds to a factor and has 
        the n_levels number of indices.

        The type of the array is object in order to allow the
        output values in terms complex objects
         - functions, tensors and fields - not only scalar 
        values.
        '''
        # get number of levels for each factor
        n_levels_list = [factor.get_n_levels() for factor in self.factor_list]
        rarray_shape = tuple([n_levels for n_levels in n_levels_list] +
                             [self.n_outputs])
        output_array = zeros(rarray_shape, dtype=object)

        # reuse the values from the cache
        #
        for run_idx, levels in enumerate(self.input_table):
            outputs = self._output_cache.get(tuple(levels), None)
            if outputs != None:
                # print 'cached value for run', run_idx
                levels_idx = self.run2levels[run_idx]
                output_array[tuple(levels_idx)] = outputs
        return output_array

    def _get_slice_n_sims(self, factor_slices):
        '''Determine the number of sims involved in the given slice.
        '''
        run_idx_list = self.levels2run[factor_slices].flatten()
        return len(run_idx_list)

    # def _get_slice_runs( self, factor_slices ):

    #---------------------------------------------------------------
    # RESULT DATA ARRAY / DICT
    #-------------------------------------------------------------------
    # cached management of the calculated data
    # parallel scheduling of computational runs
    #
    _output_cache = Dict

    def __getitem__(self, factor_slices):
        '''
        Access to the output_array using factor level indices.

        This method enables access to the values using the syntax

        output_sub_array = pstudy[ f1_level_idx, f2_level_idx, ... ]

        Here the standard numpy indices including slice and elipses can be used. 
        '''

        # map the slices within the levels2run array
        # to the indices of the expanded input_table
        #
        n_sims = self._get_slice_n_sims(factor_slices)
        progress = ProgressDialog(title='simulation progress',
                                  message='running %d simulations' % n_sims,
                                  max=n_sims,
                                  show_time=True,
                                  can_cancel=True)
        progress.open()

        run_idx_list = self.levels2run[factor_slices].flatten()
        runs_levels = self.input_table[run_idx_list]
        runs_levels_idx = self.run2levels[run_idx_list]

        # start the computation for each of the runs
        #
        sim_idx = 0
        for run_levels, run_levels_idx in zip(runs_levels, runs_levels_idx):

            # check to see if this run is already in the cache
            #
            outputs = self._output_cache.get(tuple(run_levels), None)
            if outputs == None:

                print('new simulation', sim_idx)

                # Set the factor values of the run in
                # the simulation model
                #
                for factor_name, factor, level in zip(self.factor_names,
                                                      self.factor_list,
                                                      run_levels):
                    level = factor.get_level_value(level)
                    setattr(self.sim_model, factor_name, level)

                # Perform the simulation
                #
                outputs = self.sim_model.peval()

                self.output_array[tuple(run_levels_idx)] = outputs
                self._output_cache[tuple(run_levels)] = outputs

            else:
                print('cached simulation', sim_idx)

            # let the progress bar interfere
            #
            (cont, skip) = progress.update(sim_idx)
            if not cont or skip:
                break
            sim_idx += 1

        progress.update(n_sims)
        return self.output_array[factor_slices]

    n_runs = Property(Int, depends_on='factor_dict.+levels_modified')

    @cached_property
    def _get_n_runs(self):
        return self.input_table.shape[0]

    fraction_cached = Property(Float, depends_on='factor_dict,_output_cache')

    @cached_property
    def _get_fraction_cached(self):
        return float(len(self._output_cache)) / float(len(self.run_table))

    output_table = Property(Array(object),
                            depends_on='factor_dict.+levels_modified,_output_cache')

    @cached_property
    def _get_output_table(self):
        '''
        Expand the ouput_array into two-dimensional table with
        first index representing the run and second index the output

        @todo: this can be probably simplifed by an index mapping 
             - check examples in numpy
        '''
        output_table = zeros((self.n_runs, self.n_outputs), dtype=object)
        for run_idx, levels in enumerate(self.run2levels):
            output_table[run_idx, :] = self.output_array[tuple(levels)]
        return output_table

    run_table = Property(Array(object),
                         depends_on='factor_dict.+levels_modified,_output_cache')

    @cached_property
    def _get_run_table(self):
        '''
        Glue together the input and output tables.
        '''
        return hstack([self.input_table, self.output_table])

    def clear_cache(self):
        ''' Clear the output cache '''
        self._output_cache = {}

    def save(self, file):
        pickle.dump(self, file)

    def load(self, file):
        self = pickle.load(file)

    #-------------------------------------------------------------------------
    # VIEW
    #-------------------------------------------------------------------------

    traits_view = View(
        HSplit(
            VGroup(
                VGroup(
                    Item('selected_factor_name',
                         editor=EnumEditor(name='factor_names'),
                         show_label=False),
                    Item('selected_factor@',
                         resizable=True,
                         show_label=False),
                    label='factor levels',
                    id='sim_pstudy.view_model.factor',
                    dock='tab',
                    scrollable=True,
                ),
                VGroup(
                    Item('selected_output_name',
                         editor=EnumEditor(name='output_names'),
                         show_label=False),
                    Item('selected_output@',
                         show_label=False,
                         springy=True),
                    label='vertical axis',
                    id='sim_psrudy.viewmodel.control',
                    dock='tab',
                ),
                id='sim_pstudy.viewmodel.factor',
                layout='split',
                label='plot range specification',
                dock='tab',
            ),
            VSplit(
                VGroup(
                    HGroup(Item('fraction_cached',
                                style='readonly',
                                label='cached [%]'),
                           Item('n_runs',
                                style='readonly',
                                label='number of runs')
                           ),
                    Item('run_table',
                         editor=input_table_editor,
                         show_label=False,
                         style='custom'),
                    label='run table',
                    id='sim_psrudy.viewmodel.input_table',
                    dock='tab',
                ),
                id='sim_pstudy.viewmodel.right',
            ),
            id='sim_pstudy.viewmodel.splitter',
            #group_theme = '@G',
            #item_theme  = '@B0B',
            #label_theme = '@BEA',
        ),
        id='sim_pstudy.viewmodel',
        dock='tab',
        resizable=True,
        height=0.8, width=0.8,
    )


def run():
    from .sim_model import SimModel
    sim_array = SimArray(sim_model=SimModel())
    print(sim_array.levels2run[:, 0, 0, 0])
    print(sim_array[:, 0, 0, 0])
    print(sim_array.run_table[0])
    print('array_content', sim_array.output_array[0, 0, 0, 0, :])
    sim_array.clear_cache()
    print('array_content', sim_array.output_array[0, 0, 0, 0, :])
    sim_array.configure_traits()

if __name__ == '__main__':
    run()
