'''
Created on Apr 24, 2019

Remarks to code
 - Homam please keep the line length at the maximum 80 characters
'''
import os
import string

from matplotlib.figure import Figure
from pyface.api import FileDialog
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from util.traits.editors import MPLFigureEditor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traits.api as tr
import traitsui.api as ui

from .HCFFPlot import HCFFPlot


class HCFF(tr.HasStrictTraits):
    '''High-Cycle Fatigue Filter
    '''
    
    #===========================================================================
    # Traits definitions
    #===========================================================================
    decimal = tr.Enum(',', '.')
    delimiter = tr.Str(';')
    file_csv = tr.File
    open_file_csv = tr.Button('Input file')
    skip_rows = tr.Int(4, auto_set=False, enter_set=True)
    figure = tr.Instance(Figure)
    columns_headers_list = tr.List([])
    x_axis = tr.Enum(values='columns_headers_list')
    y_axis = tr.Enum(values='columns_headers_list')
    x_axis_multiplier = tr.Enum(1, -1)
    y_axis_multiplier = tr.Enum(-1, 1)
    npy_folder_path = tr.Str
    file_name = tr.Str
    apply_filter = tr.Bool
    force_name = tr.Str('Kraft')
    peak_force_before_cycles = tr.Float(30)
    plot_creep = tr.Button
    parse_csv_to_npy = tr.Button
    plot = tr.Button
    generate_filtered_npy = tr.Button
    force_max = tr.Float(100)
    force_min = tr.Float(40)
#     plots_list = tr.List(editor=ui.SetEditor(
#         values=['kumquats', 'pomegranates', 'kiwi'],
#         can_move_all=False,
#         left_column_title='List'))
    
    #=========================================================================
    # File management
    #=========================================================================

    def _open_file_csv_fired(self):
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
        headers_array = np.array(pd.read_csv(
            self.file_csv, delimiter=self.delimiter, decimal=self.decimal, nrows=1, header=None))[0]
        for i in range(len(headers_array)):
            headers_array[i] = self.get_valid_file_name(headers_array[i])
        self.columns_headers_list = list(headers_array)
        
        """ Saving file name and path and creating NPY folder """
        dir_path = os.path.dirname(self.file_csv)
        self.npy_folder_path = os.path.join(dir_path, 'NPY')
        if os.path.exists(self.npy_folder_path) == False:
            os.makedirs(self.npy_folder_path)
            
        self.file_name = os.path.splitext(os.path.basename(self.file_csv))[0]

    #=========================================================================
    # Parameters of the filter algorithm
    #=========================================================================

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure
       
    def _parse_csv_to_npy_fired(self):
        print('Parsing csv into npy files...')
        
        for i in range(len(self.columns_headers_list)):
            column_array = np.array(pd.read_csv(
                self.file_csv, delimiter=self.delimiter, decimal=self.decimal, skiprows=self.skip_rows, usecols=[i]))
            np.save(os.path.join(self.npy_folder_path, self.file_name +
                                 '_' + self.columns_headers_list[i] + '.npy'), column_array)

        print('Finsihed parsing csv into npy files.')

    def get_valid_file_name(self, original_file_name):
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        new_valid_file_name = ''.join(
            c for c in original_file_name if c in valid_chars)
        return new_valid_file_name

    def _generate_filtered_npy_fired(self):

        # 1- Export filtered force
        force = np.load(os.path.join(self.npy_folder_path, self.file_name + '_' + self.force_name + '.npy')).flatten()
        peak_force_before_cycles_index = np.where(abs((force)) > abs(self.peak_force_before_cycles))[0][0]
        force_ascending = force[0:peak_force_before_cycles_index]
        force_rest = force[peak_force_before_cycles_index:]
        
        force_max_indices, force_min_indices = self.get_array_max_and_min_indices(force_rest)

        force_max_min_indices = np.concatenate((force_min_indices, force_max_indices))
        force_max_min_indices.sort()
        
        force_rest_filtered = force_rest[force_max_min_indices]
        force_filtered = np.concatenate((force_ascending, force_rest_filtered))
        np.save(os.path.join(self.npy_folder_path, self.file_name + '_' + self.force_name + '_filtered.npy'), force_filtered)
        
        # 2- Export filtered displacements
        # TODO I skipped time with presuming it's the first column
        for i in range(1, len(self.columns_headers_list)):
            if self.columns_headers_list[i] != str(self.force_name):
                
                disp = np.load(os.path.join(self.npy_folder_path, self.file_name + '_' + self.columns_headers_list[i] + '.npy')).flatten()
                disp_ascending = disp[0:peak_force_before_cycles_index]
                disp_rest = disp[peak_force_before_cycles_index:]
                filtered_disp = self.smooth_ascending_disp_branch(disp_ascending, disp_rest, force_max_min_indices)
                np.save(os.path.join(self.npy_folder_path, self.file_name + '_' + self.columns_headers_list[i] + '_filtered.npy'), filtered_disp)
                
        # 3- Export creep for displacements
        # Cutting unwanted max min values to get correct full cycles and remove false min/max values caused by noise
        force_max_indices_cutted, force_min_indices_cutted = self.cut_indices_in_range(force_rest,
                                                             force_max_indices,
                                                             force_min_indices,
                                                             self.force_max,
                                                             self.force_min)

        print("Cycles number= ", len(force_min_indices))
        print("Cycles number after cutting unwanted max-min range= ", len(force_min_indices_cutted))
        
        # TODO I skipped time with presuming it's the first column
        for i in range(1, len(self.columns_headers_list)):
            if self.columns_headers_list[i] != str(self.force_name):
                disp_rest_maxima = disp_rest[force_max_indices_cutted]
                disp_rest_minima = disp_rest[force_min_indices_cutted]
                np.save(os.path.join(self.npy_folder_path, self.file_name + '_' + self.columns_headers_list[i] + '_max.npy'), disp_rest_maxima)
                np.save(os.path.join(self.npy_folder_path, self.file_name + '_' + self.columns_headers_list[i] + '_min.npy'), disp_rest_minima)

        print('Filtered npy files are generated.')

    def cut_indices_in_range(self, array, max_indices, min_indices, range_upper_value, range_lower_value):
        cutted_max_indices = []
        cutted_min_indices = []
        
        for max_index in max_indices:
            if abs(array[max_index]) > abs(range_upper_value):
                cutted_max_indices.append(max_index)
        for min_index in min_indices:
            if abs(array[min_index]) < abs(range_lower_value):
                cutted_min_indices.append(min_index)
        return cutted_max_indices, cutted_min_indices
        
    def smooth_ascending_disp_branch(self, disp_ascending, disp_rest, force_extrema_indices):
        disp_ascending = savgol_filter(disp_ascending, window_length=51, polyorder=2)
        disp_rest = disp_rest[force_extrema_indices]
        disp = np.concatenate((disp_ascending, disp_rest)) 
        return disp

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
        
        # Remove subsequent max/min indices (np.greater_equal will give 1,2 for [4, 8, 8, 1])
        force_max_indices = self.remove_sequent_max_values(force_max_indices)
        force_min_indices = self.remove_sequent_min_values(force_min_indices)
        
        # If size is not equal remove the last element from the big one
        if force_max_indices.size > force_min_indices.size:
            force_max_indices = force_max_indices[:-1]
        elif force_max_indices.size < force_min_indices.size:
            force_min_indices = force_min_indices[:-1]
            
        return force_max_indices, force_min_indices
    
    def remove_sequent_max_values(self, force_max_indices):
        to_delete_from_maxima = []
        for i in range(force_max_indices.size - 1):
            if force_max_indices[i + 1] - force_max_indices[i] == 1:
                to_delete_from_maxima.append(i)
        
        force_max_indices = np.delete(force_max_indices, to_delete_from_maxima)
        return force_max_indices
        
    def remove_sequent_min_values(self, force_min_indices):
        to_delete_from_minima = []
        for i in range(force_min_indices.size - 1):
            if force_min_indices[i + 1] - force_min_indices[i] == 1:
                to_delete_from_minima.append(i)
        force_min_indices = np.delete(force_min_indices, to_delete_from_minima)
        return force_min_indices

    #===========================================================================
    # Plotting
    #===========================================================================

    def _plot_fired(self):
        
        print('Loading npy files...')

        if self.apply_filter:
            x_axis_array = float(self.x_axis_multiplier) * \
                np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.x_axis + '_filtered.npy'))
            y_axis_array = float(self.y_axis_multiplier) * \
                np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.y_axis + '_filtered.npy'))
        else:
            x_axis_array = float(self.x_axis_multiplier) * \
                np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.x_axis + '.npy'))
            y_axis_array = float(self.y_axis_multiplier) * \
                np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.y_axis + '.npy'))

        print('Plotting...')
        mpl.rcParams['agg.path.chunksize'] = 50000

        ax = self.figure.add_subplot(111)
        ax.xlabel('Displacement [mm]')
        ax.ylabel('kN')
        ax.title('Original data', fontsize=20)
        ax.plot(x_axis_array, y_axis_array, 'k', linewidth=0.8)
        self.data_changed = True

#         plt.figure()
#         plt.xlabel('Displacement [mm]')
#         plt.ylabel('kN')
#         plt.title('Original data', fontsize=20)
#         plt.plot(x_axis_array, y_axis_array, 'k', linewidth=0.8)
#
#         plt.show()
        print('Finished plotting!')
        
    def _plot_creep_fired(self):
        
        disp_max = self.x_axis_multiplier * np.load(os.path.join(self.npy_folder_path, self.file_name + '_' + self.x_axis + '_max.npy'))
        disp_min = self.x_axis_multiplier * np.load(os.path.join(self.npy_folder_path, self.file_name + '_' + self.x_axis + '_min.npy'))
        
        print('Plotting...')
        mpl.rcParams['agg.path.chunksize'] = 50000
        
        plt.figure()
        plt.xlabel('Cycles number')
        plt.ylabel('mm')
        plt.title('Fatigue creep curve', fontsize=20)
        plt.plot(np.arange(0, disp_max.size), disp_max, 'k', linewidth=0.8, color='red')
        plt.plot(np.arange(0, disp_min.size), disp_min, 'k', linewidth=0.8, color='green')
        
        plt.show()
        print('Finished plotting!')
    
    #===========================================================================
    # Configuration of the view
    #===========================================================================
    traits_view = ui.View(
        ui.HSplit(
            ui.VSplit(
                ui.HGroup(
                    ui.UItem('open_file_csv'),
                    ui.UItem('file_csv', style='readonly'),
                    label='Input data'
                ),
                ui.VGroup(
                    ui.Item('skip_rows'),
                    ui.Item('decimal'),
                    ui.Item('delimiter'),
                    label='Filter parameters'
                ),
                ui.VGroup(
                    ui.Item('parse_csv_to_npy', show_label=False),
                    ui.Item('generate_filtered_npy', show_label=False),
                    ui.Item('plot', show_label=False),
                    ui.Item('plot_creep', show_label=False)
                )
            ),
            ui.VGroup(
                ui.VGroup(
                    ui.HGroup(ui.Item('x_axis'), ui.Item('x_axis_multiplier')),
                    ui.HGroup(ui.Item('y_axis'), ui.Item('y_axis_multiplier')),
                    show_border=True,
                    label='Plotting settings'),
                ui.VGroup(
                    ui.Item('force_name'),
                    ui.HGroup(ui.Item('apply_filter'), ui.Item('peak_force_before_cycles'), show_border=True, label='Skip noise of ascending branch filter'),
                    ui.Item('force_max'),
                    ui.Item('force_min'),
#                     ui.Item('plots_list'),
                    ui.HGroup(show_border=True, label='Other filter'),
                    show_border=True,
                    label='Filters'),
            ),
            ui.UItem('figure', editor=MPLFigureEditor(),
                     resizable=True,
                     springy=True,
                     label='2d plots'),
        ),
        resizable=True,
        width=0.8,
        height=0.6
    )


if __name__ == '__main__':
    hcff = HCFF(file_csv='C:\\Users\\hspartali\\Desktop\\')
    hcff.configure_traits()

