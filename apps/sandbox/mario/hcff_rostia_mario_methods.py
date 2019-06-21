'''
Created on Apr 24, 2019

@author: rch
'''
import os

from matplotlib.figure import Figure
from pyface.api import FileDialog
from util.traits.editors import MPLFigureEditor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traits.api as tr
import traitsui.api as ui

from .something_traits import Something


class HCFF(tr.HasStrictTraits):
    '''High-Cycle Fatigue Filter
    '''

    something = tr.Instance(Something)
    decimal = tr.Enum(',', '.')
    delimiter = tr.Str(';')
    
    path_hdf5 = tr.Str('')

    def _something_default(self):
        return Something()

    #=========================================================================
    # File management
    #=========================================================================
    file_csv = tr.File

    open_file_csv = tr.Button('Input file')

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

    chunk_size = tr.Int(10000, auto_set=False, enter_set=True)

    skip_rows = tr.Int(4, auto_set=False, enter_set=True)

    # 1) use the decorator
    @tr.on_trait_change('chunk_size, skip_rows')
    def whatever_name_size_changed(self):
        print('chunk-size changed')

    # 2) use the _changed or _fired extension
    def _chunk_size_changed(self):
        print('chunk_size changed - calling the named function')

    data = tr.Array(dtype=np.float_)

    read_loadtxt_button = tr.Button()

    def _read_loadtxt_button_fired(self):
        self.data = np.loadtxt(self.file_csv, skiprows=self.skip_rows, delimiter=self.delimiter)
        print(self.data.shape)

    read_csv_button = tr.Button
    read_hdf5_button = tr.Button

    def _read_csv_button_fired(self):
        self.read_csv()
        
    def _read_hdf5_button_fired(self):
        self.read_hdf5_no_filter()

    def read_csv(self):
        '''Read the csv file and transform it to the hdf5 format.
        The output file has the same name as the input csv file
        with an extension hdf5
        '''
        path_csv = self.file_csv
        # Following splitext splits the path into a pair (root, extension)
        self.path_hdf5 = os.path.splitext(path_csv)[0] + '.hdf5'
        
        for i, chunk in enumerate(pd.read_csv(path_csv, delimiter=self.delimiter, decimal=self.decimal, skiprows=self.skip_rows, chunksize=self.chunk_size)):
            chunk_array = np.array(chunk)
            chunk_data_frame = pd.DataFrame(chunk_array, columns=['a', 'b', 'c', 'd', 'e', 'f'])
            if i == 0:
                chunk_data_frame.to_hdf(self.path_hdf5, 'all_data', mode='w', format='table')
            else:
                chunk_data_frame.to_hdf(self.path_hdf5, 'all_data', append=True)
            
    def read_hdf5_no_filter(self):

        # reading hdf files is really memory-expensive!
        force = np.array(pd.read_hdf(self.path_hdf5, columns=['b']))
        weg = np.array(pd.read_hdf(self.path_hdf5, columns=['c']))
        disp1 = np.array(pd.read_hdf(self.path_hdf5, columns=['d']))
        disp2 = np.array(pd.read_hdf(self.path_hdf5, columns=['e']))
        disp3 = np.array(pd.read_hdf(self.path_hdf5, columns=['f']))

        force = np.concatenate((np.zeros((1, 1)), force))
        weg = np.concatenate((np.zeros((1, 1)), weg))
        disp1 = np.concatenate((np.zeros((1, 1)), disp1))
        disp2 = np.concatenate((np.zeros((1, 1)), disp2))
        disp3 = np.concatenate((np.zeros((1, 1)), disp3))
        
        dir_path = os.path.dirname(self.file_csv)
        npy_folder_path = os.path.join(dir_path, 'NPY')
        if os.path.exists(npy_folder_path) == False:
            os.makedirs(npy_folder_path)
            
        file_name = os.path.splitext(os.path.basename(self.file_csv))[0]

        np.save(os.path.join(npy_folder_path, file_name + '_Force_nofilter.npy'), force)
        np.save(os.path.join(npy_folder_path, file_name + '_Displacement_machine_nofilter.npy'), weg)
        np.save(os.path.join(npy_folder_path, file_name + '_Displacement_sliding1_nofilter.npy'), disp1)
        np.save(os.path.join(npy_folder_path, file_name + '_Displacement_sliding2_nofilter.npy'), disp2)
        np.save(os.path.join(npy_folder_path, file_name + '_Displacement_crack1_nofilter.npy'), disp3)
        
        # Defining chunk size for matplotlib points visualization
        mpl.rcParams['agg.path.chunksize'] = 50000
         
        plt.subplot(111)
        plt.xlabel('Displacement [mm]')
        plt.ylabel('kN')
        plt.title('original data', fontsize=20)
        plt.plot(disp2, force, 'k')
        plt.show()
    
    figure = tr.Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    columns_headers_list = tr.List([])
    x_axis = tr.Enum(values='columns_headers_list')
    y_axis = tr.Enum(values='columns_headers_list')
    npy_folder_path = tr.Str
    file_name = tr.Str

    plot = tr.Button      
                 
    def _plot_fired(self):
        ax = self.figure.add_subplot(111)
        print('plotting figure')
        print(type(self.x_axis), type(self.y_axis))
        print(self.data[:, 1])
        print(self.data[:, self.x_axis])
        print(self.data[:, self.y_axis])
        ax.plot(self.data[:, self.x_axis], self.data[:, self.y_axis])
               
    traits_view = ui.View(
        ui.HSplit(
            ui.VSplit(
                ui.HGroup(
                    ui.UItem('open_file_csv'),
                    ui.UItem('file_csv', style='readonly'),
                    label='Input data'
                ),
                ui.VGroup(
                    ui.Item('chunk_size'),
                    ui.Item('skip_rows'),
                    ui.Item('decimal'),
                    ui.Item('delimiter'),
                    label='Filter parameters'
                ),
                ui.VGroup(
                    ui.HGroup(ui.Item('read_loadtxt_button', show_label=False),
                              ui.Item('plot', show_label=False),
                              show_border=True),
                    ui.HGroup(ui.Item('read_csv_button', show_label=False),
                              ui.Item('read_hdf5_button', show_label=False) ,
                              show_border=True)
                )
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
    name = 'CT80-39_6322_Zykl_dot'
    home_dir = os.path.expanduser('~')
    path_master = os.path.join(
        home_dir, 'Data Processing', 'CT', 'C80', 'CT80-39_6322_Zykl',
        name + '.csv'
    )
    print(path_master)
    hcff = HCFF(file_csv=path_master)
    # hcff._read_loadtxt_button_fired()
    hcff.configure_traits()

# other traits imports
