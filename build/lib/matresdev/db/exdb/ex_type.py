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
# Created on Feb 15, 2010 by: rch

# @todo: introduce the activation of filters - ironing, smoothing

import configparser
import importlib
import os
import string
import string
import sys
import zipfile

from traits.api import \
    File, \
    Array, Str, Property, cached_property, \
    Dict, Bool, provides, Float, Callable

from matresdev.db import SimDBClass
from matresdev.db.simdb import SFTPServer
from matresdev.db.simdb.simdb import simdb
import numpy as np

from .i_ex_type import \
    IExType
from .loadtxt_novalue import loadtxt_novalue


def comma2dot(c):
    '''convert float with comma separator into float with dot separator'''
    return float((str(c)).replace(",", "."))


def time2sec(date):
    '''convert time format (hh:mm:ss) to seconds (s)'''
    d_list = str(date).split()
    t_list = d_list[1].split(':')
    t_sec = int(t_list[0]) * 60 * 60 + int(t_list[1]) * \
        60 + float(comma2dot(t_list[2]))
    return t_sec


def scaledown_data(data_arr, n_avg):
    '''scaledown the nuber of rows in 'data_array' by the
    integer 'n_avg', i.e. if 'n_avg=2' reduce the number of rows in
    'data_arr' to half. The methods cuts of  up to 2*'n_avgs' rows from the end
    of the file in order to make sure that the sub arrays used for averaging have the same
    shape; in general that doesn't effect the data as the measuring is continued after rupture
    or high frequency measurements is used'''
    n_rows = data_arr.shape[0]
    n_steps = (n_rows - n_avg) / n_avg
    n_max = n_steps * n_avg
    avg_list = [data_arr[i:n_max:n_avg, :] for i in range(n_avg)]
    avg_arr = np.array(avg_list)
    return np.mean(avg_arr, 0)


@provides(IExType)
class ExType(SimDBClass):

    '''Read the data from the directory
    '''

    data_file = File

    file_ext = Str('DAT')

    def validate(self):
        '''Validate the input data return the info whether or not
         the input is valid. This is the condition for processing
         of the derived data.
        '''
        return True

    # set a flag for the view to check whether derived data is available
    #
    derived_data_available = Bool(False)

    # specify inputs
    #
    key = Property(Str, trantient=True, depends_on='data_file')

    def _get_key(self):
        return os.path.basename(self.data_file).split('.')[0]

    def _set_key(self, value):
        genkey = os.path.basename(self.data_file).split('.')[0]
        if genkey != value:
            raise KeyError('key mismatch %s != %s' % (genkey, value))

    def __setstate__(self, state, kw={}):
        if 'key' in state:
            del state['key']
        super(SimDBClass, self).__setstate__(state, **kw)

    # indicate whether the test is suitable and prepared for
    # calibration.
    ready_for_calibration = Property(Bool)

    def _get_ready_for_calibration(self):
        # return False by default
        # the subclasses shall overload this
        # and define the rules
        return False

    # specify plot templates that can be chosen for viewing
    #
    plot_templates = Dict(transient=True)

    # define processing
    #
    processed_data_array = Array('float_', transient=True)

    def process_source_data(self):
        '''process the source data and assign
        attributes to the DAT-file channel names.
        '''
        print('*** process data ***')
        self._import_processor()
        self._apply_data_reader()
        self.processed_data_array = self.data_array
        self._set_array_attribs()
        self._apply_data_processor()

    data_columns = Callable(None)
    '''Specification of the measured data columns in the data array 
    '''
    data_units = Callable(None)
    '''Specification of the measured data units in the data array 
    '''
    data_reader = Callable(None)
    '''Function reading the data into the self.data_array
    '''
    data_processor = Callable(None)
    '''Function preparing the data for evaluation - convert the measured
    data to a standard format, smooth the data, remove jumps etc. 
    '''

    def _import_processor(self):
        '''Check to see if there is a data processor in the data directory.
        The name of the processor is assumed data_processor.py.

        '''
        dp_file = os.path.join(os.path.dirname(self.data_file),
                               'data_processor.py')
        dp_modpath = os.path.join(os.path.dirname(self.data_file),
                                  'data_processor').replace(simdb.pathchar, '.')[1:]
        exdata_dir = simdb.exdata_dir
        print('dp_modpath', dp_modpath)
        print('exdata_dir', exdata_dir)
        dp_mod = dp_modpath[len(exdata_dir):]
        print('dp_mod', dp_mod)
        print('sys.path', sys.path)
        if os.path.exists(dp_file):
            mod = importlib.import_module(dp_mod)
            print('simdb-data processor used')
            if hasattr(mod, 'data_columns'):
                self.data_columns = mod.data_columns
            if hasattr(mod, 'data_units'):
                self.data_units = mod.data_units
            if hasattr(mod, 'data_processor'):
                self.data_processor = mod.data_processor
            if hasattr(mod, 'data_reader'):
                self.data_reader = mod.data_reader

    processing_done = Bool(False)

    def _apply_data_processor(self):
        '''Make a call to a test-specific data processor
        transforming the data_array to standard response variables.
        of the test setup.

        An example of data processing for a tensile test is the calculation
        of average displacement from several gauges placed on different sides
        of the specimen. 
        '''
        if self.data_processor:
            self.data_processor(self)
            self.processing_done = True

    def _apply_data_reader(self):
        if self.data_reader:
            self.data_reader(self)
        else:
            self._read_data_array()

    data_array = Array(float, transient=True)

    unit_list = Property(depends_on='data_file')

    def _get_unit_list(self):
        if self.data_units:
            return self.data_units(self)
        else:
            return self.names_and_units[1]

    factor_list = Property(depends_on='data_file')

    def _get_factor_list(self):
        if self.data_columns:
            return self.data_columns(self)
        else:
            return self.names_and_units[0]

    names_and_units = Property(depends_on='data_file')

    @cached_property
    def _get_names_and_units(self):
        ''' Extract the names and units of the measured data.
        The order of the names in the .DAT-file corresponds
        to the order of the .ASC-file.
        '''
        # for data exported into DAT and ASC-files
        file_ = open(self.data_file, 'r')
        lines = file_.read().split()
        names = []
        units = []
        for i in range(len(lines)):
            if lines[i] == '#BEGINCHANNELHEADER':
                print('names and units are defined in DAT-file')
                name = lines[i + 1].split(',')[1]
                unit = lines[i + 3].split(',')[1]
                names.append(name)
                units.append(unit)

        # for data exported into a single csv-file
        file_split = self.data_file.split('.')
        if os.path.exists(file_split[0] + '.csv'):
            file_ = open(file_split[0] + '.csv', 'r')
            header_line_1 = file_.readline().strip()
            if header_line_1.split(';')[0] == 'Datum/Uhrzeit':
                print('csv-file with header exists')
                header_line_2 = file_.readline().strip()
                names = header_line_1.split(';')
                units = header_line_2.split(';')
                names[0] = 'Bezugskanal'
                units[0] = 'sec'

        return names, units

    def _names_and_units_default(self):
        ''' Extract the names and units of the measured data.
        The order of the names in the .DAT-file corresponds
        to the order of the .ASC-file.
        '''
        # for data exported into DAT and ASC-files
        file_ = open(self.data_file, 'r')
        lines = file_.read().split()
        names = []
        units = []
        for i in range(len(lines)):
            if lines[i] == '#BEGINCHANNELHEADER':
                name = lines[i + 1].split(',')[1]
                unit = lines[i + 3].split(',')[1]
                names.append(name)
                units.append(unit)

        # for data exported into a single csv-file
        file_split = self.data_file.split('.')
        if os.path.exists(file_split[0] + '.csv'):
            file_ = open(file_split[0] + '.csv', 'r')
            header_line_1 = file_.readline()
            if header_line_1.split(';')[0] == 'Datum/Uhrzeit':
                header_line_2 = file_.readline()
                names = header_line_1.split(';')
                units = header_line_2.split(';')
                names[0] = 'Bezugskanal'
                units[0] = 'sec'
                # cut off trailing '\r\n' at end of header line
                names[-1] = names[-1][:-2]
                units[-1] = units[-1][:-2]

        print('names, units (default)', names, units)
        return names, units

    def _set_array_attribs(self):
        '''Set the measured data as named attributes defining slices into
        the processed data array.
        '''
        for i, factor in enumerate(self.factor_list):
            self.add_trait(
                factor, Array(value=self.processed_data_array[:, i],
                              transient=True))

    # ------------------

    def _read_data_array(self):
        ''' Read the experiment data.
        '''
        if os.path.exists(self.data_file):

            print('READ FILE')
            # change the file name dat with asc
            file_split = self.data_file.split('.')
            file_name = file_split[0] + '.csv'

            # for data exported into a single csv-file
            if os.path.exists(file_name):
                print('check csv-file')
                file_ = open(file_name, 'r')
                header_line_1 = file_.readline().split()

                if header_line_1[0].split(';')[0] == 'Datum/Uhrzeit':
                    print('read csv-file')
                    # for data exported into down sampled data array
                    try:
                        _data_array = np.loadtxt(file_name,
                                                 delimiter=';',
                                                 skiprows=2)
                        # reset time[sec] in order to start at 0.
                        _data_array[:0] -= _data_array[0:0]
                    except ValueError:
                        # for first column use converter method 'time2sec';
                        converters = {0: time2sec}
                        # for all other columns use converter method
                        # 'comma2dot'
                        for i in range(len(header_line_1[0].split(';')) - 1):
                            converters[i + 1] = comma2dot
                        _data_array = np.loadtxt(
                            file_name, delimiter=";", skiprows=2, converters=converters)

                        # reset time[sec] in order to start at 0.
                        _data_array[:0] -= _data_array[0:0]

                else:
                    # for data exported into DAT and ASC-files
                    # try to use loadtxt to read data file
                    try:
                        _data_array = np.loadtxt(file_name,
                                                 delimiter=';')

                    # loadtxt returns an error if the data file contains
                    # 'NOVALUE' entries. In this case use the special
                    # method 'loadtxt_novalue'
                    except ValueError:
                        _data_array = loadtxt_novalue(file_name)

            if not os.path.exists(file_name):
                file_name = file_split[0] + '.ASC'
                if not os.path.exists(file_name):
                    raise IOError('file %s does not exist' % file_name)

                # for data exported into DAT and ASC-files
                # try to use loadtxt to read data file
                try:
                    _data_array = np.loadtxt(file_name,
                                             delimiter=';')

                # loadtxt returns an error if the data file contains
                # 'NOVALUE' entries. In this case use the special
                # method 'loadtxt_novalue'
                except ValueError:
                    _data_array = loadtxt_novalue(file_name)

            self.data_array = _data_array

    data_dir = Property()
    '''Local directory path of the data file.
    '''

    def _get_data_dir(self):
        return os.path.dirname(self.data_file)

    relative_path = Property
    '''Relative path inside database structure - the path is same for experiment
    in both database structures (remote and local)
    '''

    def _get_relative_path(self):
        return self.data_dir.replace(simdb.simdb_dir, '')[1:]

    hook_up_file = Property
    '''File specifying the access to extended data.
    The cfg file is used to hook up arbitrary type
    of data stored anywhere that can be downloaded
    on demand to the local cache.
    '''

    def _get_hook_up_file(self):
        dir_path = os.path.dirname(self.data_file)
        file_name = os.path.basename(self.data_file)
        file_split = file_name.split('.')
        file_name = os.path.join(dir_path,
                                 file_split[0] + '.cfg')
        if not os.path.exists(file_name):
            file_name = ''
        return file_name

    aramis_start_offset = Property(Float, depends_on='data_file')
    '''Get time offset of aramis start specified in the hookup file.
    '''
    @cached_property
    def _get_aramis_start_offset(self):
        # hook_up an extended file if available.
        aramis_start_offset = 0.0
        if self.hook_up_file:
            config = configparser.ConfigParser()
            config.read(self.hook_up_file)
            try:
                aramis_start_offset = config.get('aramis_data',
                                                 'aramis_start_offset')
            except configparser.NoOptionError:
                pass
        return float(aramis_start_offset)

    aramis_files = Property(depends_on='data_file')
    '''Get the list of available aramis files specified in the hookup file.
    '''
    @cached_property
    def _get_aramis_files(self):
        # hook_up an extended file if available.
        aramis_files = []
        if self.hook_up_file:
            config = configparser.ConfigParser()
            config.read(self.hook_up_file)
            aramis_files = config.get(
                'aramis_data', 'aramis_files').split(',\n')
        return aramis_files

    aramis_dict = Property(depends_on='data_file')
    '''Use the last two specifiers of the aramis file name
    as a key to access the proper file.
    '''
    @cached_property
    def _get_aramis_dict(self):
        # hook_up an extended file if available.
        af_dict = {}
        for af in self.aramis_files:
            fx, fy = af.split('-')[-2:]
            af_dict[fx + '-' + fy] = af
        return af_dict

    def download_aramis_file(self, arkey):
        af = self.aramis_dict[arkey]
        af_rel_dir = os.path.join(self.relative_path, 'aramis')
        af_local_dir = os.path.join(simdb.simdb_cache_dir, af_rel_dir)
        if not os.path.exists(af_local_dir):
            os.makedirs(af_local_dir)
        try:
            s = SFTPServer(simdb.server_username, '', simdb.server_host)
            if hasattr(s, 'sftp'):
                zip_filename = af + '.zip'
                zipfile_server = os.path.join(
                    simdb.simdb_cache_remote_dir, af_rel_dir, zip_filename)

                zipfile_server = zipfile_server.replace('\\', '/')
                zipfile_local = os.path.join(af_local_dir, zip_filename)

                print('downloading', zipfile_server)
                print('destination', zipfile_local)

                s.download(zipfile_server, zipfile_local)
                s.sftp.stat(zipfile_server)
                s.close()
        except IOError as e:
            raise IOError(e)

    def uncompress_aramis_file(self, arkey):
        af = self.aramis_dict[arkey]
        af_rel_dir = os.path.join(self.relative_path, 'aramis')
        af_local_dir = os.path.join(simdb.simdb_cache_dir, af_rel_dir)
        zip_filename = af + '.zip'
        zipfile_local = os.path.join(af_local_dir, zip_filename)
        if not os.path.exists(zipfile_local):
            self.download_aramis_file(arkey)

        print('uncompressing')
        zf = zipfile.ZipFile(zipfile_local, 'r')
        zf.extractall(af_local_dir)
        zf.close()

    def get_cached_aramis_file(self, arkey):
        '''For the specified aramis resolution key check if the file
        has already been downloaded.
        '''
        af = self.aramis_dict.get(arkey, None)
        if af is None:
            print('Aramis data not available for resolution %s of the'
                  'test data\n%s' % (arkey, self.data_file))
            return None
        af_path = os.path.join(
            simdb.simdb_cache_dir, self.relative_path, 'aramis', af)

        if not os.path.exists(af_path):
            self.uncompress_aramis_file(arkey)

        return af_path
