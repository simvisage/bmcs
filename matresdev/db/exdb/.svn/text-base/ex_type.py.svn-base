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
# Created on Feb 15, 2010 by: rch

# @todo: introduce the activation of filters - ironing, smoothing

from enthought.traits.api import \
    HasTraits, Directory, List, Int, Float, Any, \
    on_trait_change, File, Constant, Instance, Trait, \
    Array, Str, Property, cached_property, WeakRef, \
    Dict, Button, Bool, Enum, Event, implements

from enthought.util.home_directory import \
    get_home_directory

from enthought.traits.ui.api import \
    View, Item, DirectoryEditor, TabularEditor, HSplit, VGroup, \
    TableEditor, EnumEditor, Handler, FileEditor, VSplit, Group

from enthought.traits.ui.table_column import \
    ObjectColumn

from enthought.traits.ui.menu import \
    OKButton, CancelButton

from enthought.traits.ui.tabular_adapter \
    import TabularAdapter

from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure

import os

import csv

from numpy import \
    array, fabs, where, copy, ones, hstack, zeros, cumsum, histogram, \
    interp

from numpy import \
    loadtxt, argmax, polyfit, poly1d, frompyfunc, dot, max

from enthought.traits.ui.table_filter \
    import EvalFilterTemplate, MenuFilterTemplate, RuleFilterTemplate, \
           EvalTableFilter

#-- Tabular Adapter Definition -------------------------------------------------

from string import replace
from os.path import exists

from loadtxt_novalue import loadtxt_novalue

#-----------------------------------------------------------------------------------
# ExDesignReader
#-----------------------------------------------------------------------------------
from enthought.traits.ui.file_dialog  \
    import open_file, FileInfo, TextInfo, ImageInfo

from enthought.traits.ui.api \
    import View, Item, TabularEditor

from enthought.traits.ui.tabular_adapter \
    import TabularAdapter

from string import split

from i_ex_type import \
    IExType

class ExType( HasTraits ):
    '''Read the data from the_ directory
    '''

    implements( IExType )

    data_file = File

    file_ext = Str( 'DAT' )

    def validate( self ):
        '''Validate the input data return the info whether or not 
         the input is valid. This is the condition for processing
         of the derived data.
        '''
        return True

    # set a flag for the view to check whether derived data is available
    #
    derived_data_available = Bool( False )

    # specify inputs
    #
    key = Property( Str, depends_on = 'data_file' )
    def _get_key( self ):
        return split( os.path.basename( self.data_file ), '.' )[0]

    # indicate whether the test is suitable and prepared for
    # calibration.
    ready_for_calibration = Property( Bool )
    def _get_ready_for_calibration( self ):
        # return False by default
        # the subclasses shall overload this 
        # and define the rules
        return False

    # specify plot templates that can be chosen for viewing
    #
    plot_templates = Dict( transient = True )

    # define processing
    #
    processed_data_array = Array( 'float_', transient = True )

    def process_source_data( self ):
        '''process the source data and assign
        attributes to the DAT-file channel names.
        '''
        print '*** process data ***'
        self._read_data_array()
        self.processed_data_array = self.data_array
        self._set_array_attribs()

    data_array = Array( float, transient = True )

    unit_list = Property( depends_on = 'data_file' )
    def _get_unit_list( self ):
        return self.names_and_units[1]

    factor_list = Property( depends_on = 'data_file' )
    def _get_factor_list( self ):
        return self.names_and_units[0]

    names_and_units = Property( depends_on = 'data_file' )
    @cached_property
    def _get_names_and_units( self ):
        ''' Extract the names and units of the measured data.
        The order of the names in the .DAT-file corresponds 
        to the order of the .ASC-file.   
        '''
        file = open( self.data_file, 'r' )
        lines = file.read().split()
        names = []
        units = []
        for i in range( len( lines ) ):
            if lines[i] == '#BEGINCHANNELHEADER':
                name = lines[i + 1].split( ',' )[1]
                unit = lines[i + 3].split( ',' )[1]
                names.append( name )
                units.append( unit )
        return names, units

    def _set_array_attribs( self ):
        '''Set the measured data as named attributes defining slices into 
        the processed data array.
        '''
        for i, factor in enumerate( self.factor_list ):
            self.add_trait( factor, Array( value = self.processed_data_array[:, i], transient = True ) )

    #------------------

    def _read_data_array( self ):
        ''' Read the experiment data. 
        '''
        if exists( self.data_file ):

            print 'READ FILE'
            # change the file name dat with asc  
            file_split = self.data_file.split( '.' )

            file_name = file_split[0] + '.csv'
            if not os.path.exists( file_name ):

                file_name = file_split[0] + '.ASC'
                if not os.path.exists( file_name ):
                    raise IOError, 'file %s does not exist' % file_name

            print 'file_name', file_name

            # try to use loadtxt to read data file
            try:
                _data_array = loadtxt( file_name,
                                       delimiter = ';' )

            # loadtxt returns an error if the data file contains
            # 'NOVALUE' entries. In this case use the special 
            # method 'loadtxt_novalue'
            except ValueError:
                _data_array = loadtxt_novalue( file_name )

            self.data_array = _data_array

