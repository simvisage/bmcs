'''
Created on Apr 24, 2019

@author: rch
'''
import os
import string

from matplotlib.figure import Figure
from pyface.api import FileDialog
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from util.traits.editors import MPLFigureEditor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traits.api as tr
import traitsui.api as ui

from .data_filtering import smooth_ascending_disp_branch


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
        headers_array = np.array(pd.read_csv(self.file_csv, delimiter=self.delimiter, decimal=self.decimal, nrows=1, header=None))[0]
        for i in range(len(headers_array)):
            headers_array[i] = self.get_valid_file_name(headers_array[i])
        self.columns_headers_list = list(headers_array)

    #=========================================================================
    # Parameters of the filter algorithm
    #=========================================================================

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure
       
    def _parse_csv_to_npy_fired(self):
        
        print('Parsing csv into npy files...')
        
        dir_path = os.path.dirname(self.file_csv)
        self.npy_folder_path = os.path.join(dir_path, 'NPY')
        if os.path.exists(self.npy_folder_path) == False:
            os.makedirs(self.npy_folder_path)
            
        self.file_name = os.path.splitext(os.path.basename(self.file_csv))[0]

        for i in range(len(self.columns_headers_list)):
            column_array = np.array(pd.read_csv(self.file_csv, delimiter=self.delimiter, decimal=self.decimal, skiprows=self.skip_rows, usecols=[i]))
            np.save(os.path.join(self.npy_folder_path, self.file_name + '_' + self.columns_headers_list[i] + '.npy'), column_array)

        print('Finsihed parsing csv into npy files.')

    def get_valid_file_name(self, original_file_name):
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        new_valid_file_name = ''.join(c for c in original_file_name if c in valid_chars)
        return new_valid_file_name
    
    def _generate_filtered_npy_fired(self):
        
        # 1- Export filtered force
        force = np.load(os.path.join(self.npy_folder_path, self.file_name + '_' + self.force_name + '.npy')).flatten()
        peak_force_before_cycles_index = np.where(abs((force)) > abs(self.peak_force_before_cycles))[0][0]
        force_ascending = force[0:peak_force_before_cycles_index]
        force_rest = force[peak_force_before_cycles_index:]

        # Extracting the local extremum values for force:
        # Check dominant sign of force
        force_positive_count = np.sum(np.array(force) >= 0)
        force_negative_count = force.size - force_positive_count

        if (force_positive_count > force_negative_count):
            force_maxima_indices = argrelmax(force_rest)[0]
            force_minima_indices = argrelmin(force_rest)[0]
        else:
            force_maxima_indices = argrelmin(force_rest)[0]
            force_minima_indices = argrelmax(force_rest)[0]
        
        force_extrema_indices = np.concatenate((force_minima_indices, force_maxima_indices))
        force_extrema_indices.sort()
        
        print("Cycles number = ", force_maxima_indices.shape)
        
        force_rest = force_rest[force_extrema_indices]
        force_filtered = np.concatenate((force_ascending, force_rest))
        np.save(os.path.join(self.npy_folder_path, self.file_name + '_' + self.force_name + '_filtered.npy'), force_filtered)
        
        # TODO I skipped time with presuming it's the first column
        # 2- Export filtered displacements
        for i in range(1, len(self.columns_headers_list)):
            if self.columns_headers_list[i] != str(self.force_name):
                
                disp = np.load(os.path.join(self.npy_folder_path, self.file_name + '_' + self.columns_headers_list[i] + '.npy')).flatten()
                disp_ascending = disp[0:peak_force_before_cycles_index]
                disp_rest = disp[peak_force_before_cycles_index:]
                filtered_disp = smooth_ascending_disp_branch(disp_ascending, disp_rest, force_extrema_indices)
                np.save(os.path.join(self.npy_folder_path, self.file_name + '_' + self.columns_headers_list[i] + '_filtered.npy'), filtered_disp)
                
                # Export creep for displacements
                disp_rest_maxima = disp_rest[force_maxima_indices]
                disp_rest_minima = disp_rest[force_minima_indices]
                disp_rest_maxima = np.concatenate((np.zeros((1)), disp_rest_maxima))
                disp_rest_minima = np.concatenate((np.zeros((1)), disp_rest_minima))
                np.save(os.path.join(self.npy_folder_path, self.file_name + '_' + self.columns_headers_list[i] + '_max.npy'), disp_rest_maxima)
                np.save(os.path.join(self.npy_folder_path, self.file_name + '_' + self.columns_headers_list[i] + '_min.npy'), disp_rest_minima)

        print('Filtered npy files are generated.')
                              
    def _plot_fired(self):
        
        print('Loading npy files...')
        
        if self.apply_filter:
            x_axis_array = self.x_axis_multiplier * np.load(os.path.join(self.npy_folder_path, self.file_name + '_' + self.x_axis + '_filtered.npy'))
            y_axis_array = self.y_axis_multiplier * np.load(os.path.join(self.npy_folder_path, self.file_name + '_' + self.y_axis + '_filtered.npy'))
        else:
            x_axis_array = self.x_axis_multiplier * np.load(os.path.join(self.npy_folder_path, self.file_name + '_' + self.x_axis + '.npy'))
            y_axis_array = self.y_axis_multiplier * np.load(os.path.join(self.npy_folder_path, self.file_name + '_' + self.y_axis + '.npy'))
        
        print('Plotting...')
        mpl.rcParams['agg.path.chunksize'] = 50000
        
        plt.figure()
        plt.xlabel('Displacement [mm]')
        plt.ylabel('kN')
        plt.title('Original data', fontsize=20)
        plt.plot(x_axis_array, y_axis_array, 'k', linewidth=0.8)
        
        plt.show()
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

