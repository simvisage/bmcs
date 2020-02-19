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
# Created on Aug 3, 2009 by: rch
#
# How to introduce ExperimentType class - as a class corresponding to SimModel
#
# - this class gathers the information about the inputs and outputs
#   of the object.
#
# - Each directory contains the `ex_type.cls' file giving the name of hte
#   ExperimentType subclass defining the inputs and outputs of the experiment
#   and further derived data needed for processing of experiments.
#
# - The ExperimentDB has a root directory and starts by scanning the
#   sub-directories
#   for the `ex_type.cls' files. It verifies that the ExTypes are defined
#   and known
#   classes. Then, the mapping between a class and between the directories
#   is established. Typically, there is a data pool in the home directory
#   or a network file system accessible.
#
# - The result of the scanning procedure
#   is list of data directories available for
#   each experiment type. Typically, data from a single treatment are grouped
#   in the single directory, but this does not necessarily need to be the case.
#   Therefore, the grouping is done independently on the data based on
#   the values of the input factors.
#
#   Tasks: define the classes
#     - ETCompositeTensileTest
#     - ETPlateTest
#     - ETPullOutTest,
#     - ETYarnTensileTest
#
#   They inherit from class ExType specifying the inputs and outputs.
#   They also specify
#   the default association of tracers to the x- and y- axis (plot templates).
#   Further, grouping
#   of equivalent values can be provided. The values of the inputs are stored
#   in the directory in the ExType.db file using the pickle format.
#
#   The ExTypeView class communicates with the particular ExType class.
#   It should be able
#   to accommodate several curves within the plotting window.
#   It should be possible to
#   select multiple runs to be plotted.
#

# - Database of material components is stored in a separate directory tree.
#   The identification
#   of the material components is provided using a universal key
#   for a material component The component must be declared either as a matrix
#   or reinforcement. It must be globally accessible from within
#   the ExpTools and SimTools.
#

import os
from os.path import join
import pickle

from traits.api import \
    HasTraits, \
    on_trait_change, File, Instance, Trait, \
    Property, cached_property, \
    Bool, Event, provides
from traitsui.api import \
    View, Item, \
    FileEditor
from traitsui.menu import \
    OKButton, CancelButton

import apptools.persistence.state_pickler as spickle
from matresdev.db.simdb import simdb
from util.find_class import _find_class

from .ex_type import \
    ExType
from .i_ex_run import \
    IExRun
from .i_ex_type import \
    IExType


data_file_editor = FileEditor(filter=['*.DAT'])

# which pickle format to use
#
pickle_modes = {'pickle': dict(load=pickle.load,
                               dump=pickle.dump,
                               ext='.pickle'),
                'spickle': dict(load=spickle.load_state,
                                dump=spickle.dump,
                                ext='.spickle')}


@provides(IExRun)
class ExRun(HasTraits):

    '''Read the data from the DAT file containing the measured data.

    The data is described in semicolon-separated
    csv file providing the information about
    data parameters.

    '''

    data_file = File
    '''File containing the association between the factor combinations
    and data files having the data.
    '''

    pickle = Trait('pickle', pickle_modes)
    '''Pickle object as a property in order to be able to switch between
    '''

    ex_type_file_name = Property(depends_on='data_file')
    '''Derived  path to the file specifying the type of the experiment,
    the ex_type.cls file is stored in the same directory.
    '''
    @cached_property
    def _get_ex_type_file_name(self):
        dir_path = os.path.dirname(self.data_file)
        file_name = os.path.join(dir_path, 'ex_type.cls')
        return file_name

    pickle_file_name = Property(depends_on='data_file')
    '''Derived path to the pickle file storing the input data and derived output
    data associated with the experiment run.
    '''
    @cached_property
    def _get_pickle_file_name(self):
        dir_path = os.path.dirname(self.data_file)
        file_name = os.path.basename(self.data_file)
        file_split = file_name.split('.')
        file_name = os.path.join(dir_path,
                                 file_split[0] + self.pickle_['ext'])
        return file_name

    ex_type = Instance(IExType)
    '''Instance of the specialized experiment type with the particular
    inputs and data derived outputs.
    '''

    def _ex_type_default(self):
        return ExType()

    def __init__(self, data_file, **kw):
        '''Initialization: the ex_run is defined by the
        data_file. The additional data - inputs and derived outputs
        are stored in the data_file.pickle. If this file exists,
        the exrun is constructed from this data file.
        '''
        super(ExRun, self).__init__(**kw)
        self.data_file = data_file
        read_ok = False

        if os.path.exists(self.pickle_file_name):
            print('PICKLE FILE EXISTS %s' % self.pickle_file_name)
            file_ = open(self.pickle_file_name, 'r')
            try:
                self.ex_type = self.pickle_['load'](file_)
                self.unsaved = False
                read_ok = True
            except EOFError:
                read_ok = False

            file_.close()
            self.ex_type.data_file = self.data_file

            # In case that the code for processing data
            # has changed since the last dump - run
            # the processing anew. This can be skipped
            # if the code is finished.
            #
            # @todo - the refreshing of the process data
            # should be possible for all instances at once
            # this would avoid the - on-click-based refreshing
            # of the data performed here.
            #
            self.ex_type.process_source_data()
            self.ex_type.derived_data_available = True
            print('*** derived data available ***')

        if not read_ok and os.path.exists(self.ex_type_file_name):

            print('PICKLE FILE DOES NOT EXIST')

            f = open(self.ex_type_file_name, 'r')
            ex_type_klass = f.read().split('\n')[0]  # use trim here
            f.close()
            theClass = _find_class(ex_type_klass)
            if theClass is None:
                raise TypeError('class %s not found for file %s' %
                                (ex_type_klass, self.ex_type_file_name))
            self.ex_type = theClass(data_file=self.data_file)
            self.unsaved = True
            read_ok = True

            self.ex_type.process_source_data()

        if not read_ok:
            raise ValueError('Cannot instantiate ExType using %s' % data_file)

    def save_pickle(self):
        '''Store the current state of the ex_run.
        '''
        file_ = open(self.pickle_file_name, 'w')
        self.pickle_['dump'](self.ex_type, file_)
        file_.close()
        self.unsaved = False

    # Event to keep track of changes in the ex_type instance.
    # It is defined in order to inform the views about a change
    # in some input variable.
    #
    change_event = Event

    # Boolean variable set to true if the object has been changed.
    # keep track of changes in the ex_type instance. This variable
    # indicates that the run is unsaved. The change may be later
    # canceled or confirmed.
    #
    unsaved = Bool(transient=True)

    @on_trait_change('ex_type.input_change')
    def _set_changed_state(self):
        print('*** received input change event ***''')
        self.change_event = True
        self.unsaved = True

    # Boolean property indicating whether the run is suitable and prepared
    # for calibration of models.
    ready_for_calibration = Property(Bool)

    def _get_ready_for_calibration(self):

        # it must have a pickle data containing the values
        # of the test parameters
        #
        return (os.path.exists(self.ex_type_file_name) and
                self.ex_type.ready_for_calibration)

    # --------------------------------------------------------------------------
    # View specification
    # --------------------------------------------------------------------------
    view_traits = View(
        Item('ex_type@', show_label=False,
             resizable=True,
             label='experiment type'
             ),
        resizable=True,
        scrollable=True,
        # title = 'Data reader',
        id='simexdb.exrun',
        dock='tab',
        buttons=[OKButton, CancelButton],
        height=0.8,
        width=0.8)


if __name__ == '__main__':
    test_file = join(simdb.exdata_dir, 'tensile_tests', 'TT-9u',
                     'TT06-9u-V1.DAT')
#     test_file = join(simdb.exdata_dir, 'plate_tests', 'PT-10a',
#                               'PT10-10a.DAT')
    exrun = ExRun(test_file)
    exrun.configure_traits(view='view_traits')
    print('processed data', exrun.ex_type.data_array.shape)
    print('ex_type file name', exrun.ex_type_file_name)
    print('ex_type', exrun.ex_type)
    print('pickle_file_name', exrun.pickle_file_name)

    # exrun.save_pickle()
    # exrun = ExRun( '/home/rch/sim_data/sim_exdb/ex_composite_tensile_test/TT08-7a-V2.DAT' )
    # exrun.configure_traits()
