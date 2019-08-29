'''
Created on Apr 24, 2019

@author: rch, Homam Spartali

Note: To use this tool, the csv file should have the columns headers in
the first row. Also the time column should be the first column.

'''
import os
import string

from matplotlib.figure import Figure
from pyface.api import FileDialog, MessageDialog
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from util.traits.editors import MPLFigureEditor

import matplotlib as mpl
import numpy as np
import pandas as pd
import traits.api as tr
import traitsui.api as ui
from traitsui.extras.checkbox_column \
    import CheckboxColumn

average_columns_editor = ui.TableEditor(
    sortable=False,
    configurable=False,
    auto_size=False,
    columns=[CheckboxColumn(name='selected', label='Select',
                            width=0.12),
             ui.ObjectColumn(name='column_name', editable=False, width=0.24,
                             horizontal_alignment='left')])


class Column(tr.HasStrictTraits):
    column_name = tr.Str
    selected = tr.Bool(False)


class ColumnsAverage(tr.HasStrictTraits):
    columns = tr.List(Column)

    # Trait view definitions:
    traits_view = ui.View(
        ui.Item('columns',
                show_label=False,
                editor=average_columns_editor
                ),
        buttons=[ui.OKButton, ui.CancelButton],
        title='Select arrays to be averaged',
        width=0.15,
        height=0.3,
        resizable=True
    )


class HCFF(tr.HasStrictTraits):
    '''High-Cycle Fatigue Filter
    '''

    #=========================================================================
    # Traits definitions
    #=========================================================================
    decimal = tr.Enum(',', '.')
    delimiter = tr.Str(';')
    records_per_second = tr.Float(100)
    file_csv = tr.File
    open_file_csv = tr.Button('Input file')
    skip_rows = tr.Int(4, auto_set=False, enter_set=True)
    columns_headers_list = tr.List([])
    x_axis = tr.Enum(values='columns_headers_list')
    y_axis = tr.Enum(values='columns_headers_list')
    x_axis_multiplier = tr.Enum(1, -1)
    y_axis_multiplier = tr.Enum(-1, 1)
    npy_folder_path = tr.Str
    file_name = tr.Str
    apply_filters = tr.Bool
    force_name = tr.Str('Kraft')
    peak_force_before_cycles = tr.Float(30)
    plots_num = tr.Enum(1, 2, 3, 4, 6, 9)
    plot_list = tr.List()
    add_plot = tr.Button
    add_creep_plot = tr.Button(desc='Creep plot of X axis array')
    clear_plot = tr.Button
    parse_csv_to_npy = tr.Button
    generate_filtered_npy = tr.Button
    add_columns_average = tr.Button
    force_max = tr.Float(100)
    force_min = tr.Float(40)
    min_cycle_force_range = tr.Float(50)
    cutting_method = tr.Enum(
        'Define min cycle range(force difference)', 'Define Max, Min')
    columns_to_be_averaged = tr.List

    figure = tr.Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    #=========================================================================
    # File management
    #=========================================================================

    def _open_file_csv_fired(self):

        self.reset()

        """ Handles the user clicking the 'Open...' button.
        """
        extns = ['*.csv', ]  # seems to handle only one extension...
        wildcard = '|'.join(extns)

        dialog = FileDialog(title='Select text file',
                            action='open', wildcard=wildcard,
                            default_path=self.file_csv)
        dialog.open()
        self.file_csv = dialog.path

        """ Filling x_axis and y_axis with values """
        headers_array = np.array(
            pd.read_csv(
                self.file_csv, delimiter=self.delimiter, decimal=self.decimal,
                nrows=1, header=None
            )
        )[0]
        for i in range(len(headers_array)):
            headers_array[i] = self.get_valid_file_name(headers_array[i])
        self.columns_headers_list = list(headers_array)

        """ Saving file name and path and creating NPY folder """
        dir_path = os.path.dirname(self.file_csv)
        self.npy_folder_path = os.path.join(dir_path, 'NPY')
        if os.path.exists(self.npy_folder_path) == False:
            os.makedirs(self.npy_folder_path)

        self.file_name = os.path.splitext(os.path.basename(self.file_csv))[0]

    def _parse_csv_to_npy_fired(self):
        print('Parsing csv into npy files...')

        for i in range(len(self.columns_headers_list)):
            column_array = np.array(pd.read_csv(
                self.file_csv, delimiter=self.delimiter, decimal=self.decimal,
                skiprows=self.skip_rows, usecols=[i]))

            """ TODO! Create time array supposing the it's column is the
            first one in the file and that we have 100 reads in 1 second """
            if i == 0:
                column_array = np.arange(start=0.0,
                                         stop=len(column_array) /
                                         self.records_per_second,
                                         step=1.0 / self.records_per_second)

            np.save(os.path.join(self.npy_folder_path, self.file_name +
                                 '_' + self.columns_headers_list[i] + '.npy'),
                    column_array)

        """ Exporting npy arrays of averaged columns """
        for columns_names in self.columns_to_be_averaged:
            temp = np.zeros((1))
            for column_name in columns_names:
                temp = temp + np.load(os.path.join(self.npy_folder_path,
                                                   self.file_name + '_' + column_name + '.npy')).flatten()
            avg = temp / len(columns_names)

            avg_suffex = 'avg_' + '_'.join(columns_names)
            np.save(os.path.join(self.npy_folder_path, self.file_name +
                                 '_' + avg_suffex + '.npy'), avg)

            self.columns_headers_list.append(avg_suffex)

        print('Finsihed parsing csv into npy files.')

    def get_valid_file_name(self, original_file_name):
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        new_valid_file_name = ''.join(
            c for c in original_file_name if c in valid_chars)
        return new_valid_file_name

    def _clear_plot_fired(self):
        self.figure.clear()
        self.plot_list = []
        self.data_changed = True

    def _add_columns_average_fired(self):
        columns_average = ColumnsAverage()
        for name in self.columns_headers_list:
            columns_average.columns.append(Column(column_name=name))

        # kind='modal' pauses the implementation until the window is closed
        columns_average.configure_traits(kind='modal')

        columns_to_be_averaged_temp = []
        for i in columns_average.columns:
            if i.selected:
                columns_to_be_averaged_temp.append(i.column_name)

        if columns_to_be_averaged_temp:  # If it's not empty
            self.columns_to_be_averaged.append(columns_to_be_averaged_temp)

    def _generate_filtered_npy_fired(self):

        # 1- Export filtered force
        force = np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.force_name + '.npy')).flatten()
        peak_force_before_cycles_index = np.where(
            abs((force)) > abs(self.peak_force_before_cycles))[0][0]
        force_ascending = force[0:peak_force_before_cycles_index]
        force_rest = force[peak_force_before_cycles_index:]

        force_max_indices, force_min_indices = self.get_array_max_and_min_indices(
            force_rest)

        force_max_min_indices = np.concatenate(
            (force_min_indices, force_max_indices))
        force_max_min_indices.sort()

        force_rest_filtered = force_rest[force_max_min_indices]
        force_filtered = np.concatenate((force_ascending, force_rest_filtered))
        np.save(os.path.join(self.npy_folder_path, self.file_name +
                             '_' + self.force_name + '_filtered.npy'), force_filtered)

        # 2- Export filtered displacements
        # TODO I skipped time with presuming it's the first column
        for i in range(1, len(self.columns_headers_list)):
            if self.columns_headers_list[i] != str(self.force_name):

                disp = np.load(os.path.join(self.npy_folder_path, self.file_name +
                                            '_' + self.columns_headers_list[i] + '.npy')).flatten()
                disp_ascending = disp[0:peak_force_before_cycles_index]
                disp_rest = disp[peak_force_before_cycles_index:]
                disp_ascending = savgol_filter(
                    disp_ascending, window_length=51, polyorder=2)
                disp_rest_filtered = disp_rest[force_max_min_indices]
                filtered_disp = np.concatenate(
                    (disp_ascending, disp_rest_filtered))
                np.save(os.path.join(self.npy_folder_path, self.file_name + '_' +
                                     self.columns_headers_list[i] + '_filtered.npy'), filtered_disp)

        # 3- Export creep for displacements
        # Cutting unwanted max min values to get correct full cycles and remove
        # false min/max values caused by noise
        if self.cutting_method == "Define Max, Min":
            force_max_indices_cutted, force_min_indices_cutted = \
                self.cut_indices_of_min_max_range(force_rest,
                                                  force_max_indices,
                                                  force_min_indices,
                                                  self.force_max,
                                                  self.force_min)
        elif self.cutting_method == "Define min cycle range(force difference)":
            force_max_indices_cutted, force_min_indices_cutted = \
                self.cut_indices_of_defined_range(force_rest,
                                                  force_max_indices,
                                                  force_min_indices,
                                                  self.min_cycle_force_range)

        print("Cycles number= ", len(force_min_indices))
        print("Cycles number after cutting fake cycles because of noise= ",
              len(force_min_indices_cutted))

        # TODO I skipped time with presuming it's the first column
        for i in range(1, len(self.columns_headers_list)):
            array = np.load(os.path.join(self.npy_folder_path, self.file_name +
                                         '_' + self.columns_headers_list[i] + '.npy')).flatten()
            array_rest = array[peak_force_before_cycles_index:]
            array_rest_maxima = array_rest[force_max_indices_cutted]
            array_rest_minima = array_rest[force_min_indices_cutted]
            np.save(os.path.join(self.npy_folder_path, self.file_name + '_' +
                                 self.columns_headers_list[i] + '_max.npy'), array_rest_maxima)
            np.save(os.path.join(self.npy_folder_path, self.file_name + '_' +
                                 self.columns_headers_list[i] + '_min.npy'), array_rest_minima)

        print('Filtered npy files are generated.')

    def cut_indices_of_min_max_range(self, array, max_indices, min_indices, range_upper_value, range_lower_value):
        cutted_max_indices = []
        cutted_min_indices = []

        for max_index in max_indices:
            if abs(array[max_index]) > abs(range_upper_value):
                cutted_max_indices.append(max_index)
        for min_index in min_indices:
            if abs(array[min_index]) < abs(range_lower_value):
                cutted_min_indices.append(min_index)
        return cutted_max_indices, cutted_min_indices

    def cut_indices_of_defined_range(self, array, max_indices, min_indices, range_):
        cutted_max_indices = []
        cutted_min_indices = []

        for max_index, min_index in zip(max_indices, min_indices):
            if abs(array[max_index] - array[min_index]) > range_:
                cutted_max_indices.append(max_index)
                cutted_min_indices.append(min_index)

        return cutted_max_indices, cutted_min_indices

    def get_array_max_and_min_indices(self, input_array):

        # Checking dominant sign
        positive_values_count = np.sum(np.array(input_array) >= 0)
        negative_values_count = input_array.size - positive_values_count

        # Getting max and min indices
        if (positive_values_count > negative_values_count):
            force_max_indices = argrelextrema(input_array, np.greater_equal)[0]
            force_min_indices = argrelextrema(input_array, np.less_equal)[0]
        else:
            force_max_indices = argrelextrema(input_array, np.less_equal)[0]
            force_min_indices = argrelextrema(input_array, np.greater_equal)[0]

        # Remove subsequent max/min indices (np.greater_equal will give 1,2 for
        # [4, 8, 8, 1])
        force_max_indices = self.remove_subsequent_max_values(
            force_max_indices)
        force_min_indices = self.remove_subsequent_min_values(
            force_min_indices)

        # If size is not equal remove the last element from the big one
        if force_max_indices.size > force_min_indices.size:
            force_max_indices = force_max_indices[:-1]
        elif force_max_indices.size < force_min_indices.size:
            force_min_indices = force_min_indices[:-1]

        return force_max_indices, force_min_indices

    def remove_subsequent_max_values(self, force_max_indices):
        to_delete_from_maxima = []
        for i in range(force_max_indices.size - 1):
            if force_max_indices[i + 1] - force_max_indices[i] == 1:
                to_delete_from_maxima.append(i)

        force_max_indices = np.delete(force_max_indices, to_delete_from_maxima)
        return force_max_indices

    def remove_subsequent_min_values(self, force_min_indices):
        to_delete_from_minima = []
        for i in range(force_min_indices.size - 1):
            if force_min_indices[i + 1] - force_min_indices[i] == 1:
                to_delete_from_minima.append(i)
        force_min_indices = np.delete(force_min_indices, to_delete_from_minima)
        return force_min_indices

    #=========================================================================
    # Plotting
    #=========================================================================

    plot_list_current_elements_num = tr.Int(0)

    def _plot_list_changed(self):
        if len(self.plot_list) > self.plot_list_current_elements_num:
            self.plot_list_current_elements_num = len(self.plot_list)

    data_changed = tr.Event

    def _add_plot_fired(self):

        if (len(self.plot_list) >= self.plots_num):
            dialog = MessageDialog(
                title='Attention!', message='Max plots number is {}'.format(self.plots_num))
            dialog.open()
            return

        print('Loading npy files...')

        if self.apply_filters:
            x_axis_name = self.x_axis + '_filtered'
            y_axis_name = self.y_axis + '_filtered'
            x_axis_array = self.x_axis_multiplier * \
                np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.x_axis + '_filtered.npy'))
            y_axis_array = self.y_axis_multiplier * \
                np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.y_axis + '_filtered.npy'))
        else:
            x_axis_name = self.x_axis
            y_axis_name = self.y_axis
            x_axis_array = self.x_axis_multiplier * \
                np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.x_axis + '.npy'))
            y_axis_array = self.y_axis_multiplier * \
                np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.y_axis + '.npy'))

        print('Adding Plot...')
        mpl.rcParams['agg.path.chunksize'] = 50000

        ax = self.apply_new_subplot()

        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('Force [kN]')
        ax.set_title('Original data', fontsize=20)
        ax.plot(x_axis_array, y_axis_array, 'k',
                linewidth=1.5, color=np.random.rand(3,), label=self.file_name)

        ax.legend()

        self.plot_list.append('{}, {}'.format(x_axis_name, y_axis_name))
        self.data_changed = True
        print('Finished adding plot!')

    def apply_new_subplot(self):
        plt = self.figure
        if (self.plots_num == 1):
            return plt.add_subplot(1, 1, 1)
        elif (self.plots_num == 2):
            plot_location = int('12' + str(len(self.plot_list) + 1))
            return plt.add_subplot(plot_location)
        elif (self.plots_num == 3):
            plot_location = int('13' + str(len(self.plot_list) + 1))
            return plt.add_subplot(plot_location)
        elif (self.plots_num == 4):
            plot_location = int('22' + str(len(self.plot_list) + 1))
            return plt.add_subplot(plot_location)
        elif (self.plots_num == 6):
            plot_location = int('23' + str(len(self.plot_list) + 1))
            return plt.add_subplot(plot_location)
        elif (self.plots_num == 9):
            plot_location = int('33' + str(len(self.plot_list) + 1))
            return plt.add_subplot(plot_location)

    def _add_creep_plot_fired(self):

        if (len(self.plot_list) >= self.plots_num):
            dialog = MessageDialog(
                title='Attention!', message='Max plots number is {}'.format(self.plots_num))
            dialog.open()
            return

        disp_max = self.x_axis_multiplier * \
            np.load(os.path.join(self.npy_folder_path,
                                 self.file_name + '_' + self.x_axis + '_max.npy'))
        disp_min = self.x_axis_multiplier * \
            np.load(os.path.join(self.npy_folder_path,
                                 self.file_name + '_' + self.x_axis + '_min.npy'))

        print('Adding creep plot...')
        mpl.rcParams['agg.path.chunksize'] = 50000

        ax = self.apply_new_subplot()

        ax.set_xlabel('Cycles number')
        ax.set_ylabel('mm')
        ax.set_title('Fatigue creep curve', fontsize=20)

        ax.plot(np.arange(0, disp_max.size), disp_max,
                'k', linewidth=1.2, color='red', label='Max loading level')
        ax.plot(np.arange(0, disp_min.size), disp_min,
                'k', linewidth=1.2, color='green', label='Min loading level')

        ax.legend()

        self.plot_list.append('Creep: {}, {}'.format(self.x_axis, self.y_axis))
        self.data_changed = True

        print('Finished adding creep plot!')

    def reset(self):
        self.delimiter = ';'
        self.skip_rows = 4
        self.columns_headers_list = []
        self.npy_folder_path = ''
        self.file_name = ''
        self.apply_filters = False
        self.force_name = 'Kraft'
        self.plot_list = []
        self.columns_to_be_averaged = []
    #=========================================================================
    # Configuration of the view
    #=========================================================================

    traits_view = ui.View(
        ui.HSplit(
            ui.VSplit(
                ui.HGroup(
                    ui.UItem('open_file_csv'),
                    ui.UItem('file_csv', style='readonly'),
                    label='Input data'
                ),
                ui.Item('add_columns_average', show_label=False),
                ui.VGroup(
                    ui.Item('records_per_second'),
                    ui.Item('skip_rows'),
                    ui.Item('decimal'),
                    ui.Item('delimiter'),
                    ui.Item('parse_csv_to_npy', show_label=False),
                    label='Filter parameters'
                ),
                ui.VGroup(
                    ui.HGroup(ui.Item('plots_num'), ui.Item('clear_plot')),
                    ui.HGroup(ui.Item('x_axis'), ui.Item('x_axis_multiplier')),
                    ui.HGroup(ui.Item('y_axis'), ui.Item('y_axis_multiplier')),
                    ui.HGroup(ui.Item('add_plot', show_label=False),
                              ui.Item('apply_filters')),
                    ui.HGroup(ui.Item('add_creep_plot', show_label=False)),
                    ui.Item('plot_list'),
                    show_border=True,
                    label='Plotting settings'),
            ),
            ui.VGroup(
                ui.Item('force_name'),
                ui.HGroup(ui.Item('peak_force_before_cycles'),
                          show_border=True, label='Skip noise of ascending branch:'),
                #                     ui.Item('plots_list'),
                ui.VGroup(ui.Item('cutting_method'),
                          ui.VGroup(ui.Item('force_max'),
                                    ui.Item('force_min'),
                                    label='Max, Min:',
                                    show_border=True,
                                    enabled_when='cutting_method == "Define Max, Min"'),
                          ui.VGroup(ui.Item('min_cycle_force_range'),
                                    label='Min cycle force range:',
                                    show_border=True,
                                    enabled_when='cutting_method == "Define min cycle range(force difference)"'),
                          show_border=True,
                          label='Cut fake cycles for creep:'),
                ui.Item('generate_filtered_npy', show_label=False),
                show_border=True,
                label='Filters'
            ),
            ui.UItem('figure', editor=MPLFigureEditor(),
                     resizable=True,
                     springy=True,
                     width=0.8,
                     label='2d plots'),
        ),
        title='HCFF Filter',
        resizable=True,
        width=0.85,
        height=0.7

    )


if __name__ == '__main__':
    hcff = HCFF(file_csv='D:\\CSV files')

    hcff.configure_traits()
