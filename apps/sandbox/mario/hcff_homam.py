'''
Created on Apr 24, 2019

@author: rch
'''
import os
import string

from matplotlib.figure import Figure
from pyface.api import FileDialog, MessageDialog
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from util.traits.editors import MPLFigureEditor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traits.api as tr
import traitsui.api as ui

from .hcff_filters.hcff_filter import HCFFilter, HCFFChild, HCFFParent


class PlotOptions(tr.HasTraits):
    columns_headers_list = tr.List([])
    x_axis = tr.Enum(values='columns_headers_list')
    y_axis = tr.Enum(values='columns_headers_list')
    x_axis_multiplier = tr.Enum(1, -1)
    y_axis_multiplier = tr.Enum(-1, 1)
    plot = tr.Button

    view = ui.View(
        ui.HGroup(ui.Item('x_axis'), ui.Item('x_axis_multiplier')),
        ui.HGroup(ui.Item('y_axis'), ui.Item('y_axis_multiplier')),
        ui.Item('plot', show_label=False)
    )


class FileImportManager(tr.HasTraits):
    file_csv = tr.File
    open_file_csv = tr.Button('Input file')
    decimal = tr.Enum(',', '.')
    delimiter = tr.Str(';')
    skip_rows = tr.Int(4, auto_set=False, enter_set=True)
    columns_headers_list = tr.List([])

    parse_csv_to_npy = tr.Button

    view = ui.View(ui.VGroup(
        ui.HGroup(
            ui.UItem('open_file_csv'),
            ui.UItem('file_csv', style='readonly'),
        ),
        ui.Item('skip_rows'),
        ui.Item('decimal'),
        ui.Item('delimiter'),
        ui.Item('parse_csv_to_npy', show_label=False),
    ))

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

        """ Fill columns_headers_list """
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

    def get_valid_file_name(self, original_file_name):
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        new_valid_file_name = ''.join(
            c for c in original_file_name if c in valid_chars)
        return new_valid_file_name

    def _parse_csv_to_npy_fired(self):
        print('Parsing csv into npy files...')

        for i in range(len(self.columns_headers_list)):
            column_array = np.array(pd.read_csv(
                self.file_csv, delimiter=self.delimiter, decimal=self.decimal, skiprows=self.skip_rows, usecols=[i]))
            np.save(os.path.join(self.npy_folder_path, self.file_name +
                                 '_' + self.columns_headers_list[i] + '.npy'), column_array)

        print('Finsihed parsing csv into npy files.')


class HCFFRoot(HCFFParent):

    #name = tr.Str('root')

    import_manager = tr.Instance(FileImportManager, ())

    plot_options = tr.List(PlotOptions)

    output_npy = tr.Property()

    tree_view = ui.View(
        ui.VGroup(
            ui.Item('import_manager', style='custom', show_label=False)
        )
    )

    traits_view = tree_view


tree_editor = ui.TreeEditor(
    orientation='vertical',
    nodes=[
        # The first node specified is the top level one
        ui.TreeNode(node_for=[HCFFRoot],
                    auto_open=True,
                    children='filters',
                    label='=HCF',
                    #                    view=ui.View()  # Empty view
                    ),
        ui.TreeNode(node_for=[HCFFilter],
                    auto_open=True,
                    children='filters',
                    label='name',
                    #                    view=ui.View()
                    ),
        ui.TreeNode(node_for=[PlotOptions],
                    auto_open=True,
                    children='',
                    label='=Plot Options',
                    view='view'
                    )
    ]
)


class HCFF2(tr.HasStrictTraits):
    '''High-Cycle Fatigue Filter
    '''

    hcf = tr.Instance(HCFFRoot)

    def _hcf_default(self):
        return HCFFRoot(import_manager=FileImportManager())

    figure = tr.Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    traits_view = ui.View(
        ui.HSplit(
            ui.Item(name='hcf',
                    editor=tree_editor,
                    show_label=False,
                    width=0.3
                    ),
            ui.UItem('figure', editor=MPLFigureEditor(),
                     resizable=True,
                     springy=True,
                     label='2d plots')
        ),
        title='HCF Filter',
        resizable=True,
        width=0.6,
        height=0.6
    )


class HCFF(tr.HasStrictTraits):
    '''High-Cycle Fatigue Filter
    '''

    #=========================================================================
    # Traits definitions
    #=========================================================================
    decimal = tr.Enum(',', '.')
    delimiter = tr.Str(';')
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
    plot = tr.Button
    add_plot = tr.Button
    add_creep_plot = tr.Button
    parse_csv_to_npy = tr.Button
    generate_filtered_npy = tr.Button
    add_columns_average = tr.Button
    force_max = tr.Float(100)
    force_min = tr.Float(40)

    figure = tr.Instance(Figure)

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

#     def _add_columns_average_fired(self):
#         columns_average = ColumnsAverage(
#             columns_names=self.columns_headers_list)
#         # columns_average.set_columns_headers_list(self.columns_headers_list)
#         columns_average.configure_traits()

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
                disp_rest = disp_rest[force_max_min_indices]
                filtered_disp = np.concatenate((disp_ascending, disp_rest))
                np.save(os.path.join(self.npy_folder_path, self.file_name + '_' +
                                     self.columns_headers_list[i] + '_filtered.npy'), filtered_disp)

        # 3- Export creep for displacements
        # Cutting unwanted max min values to get correct full cycles and remove
        # false min/max values caused by noise
        force_max_indices_cutted, force_min_indices_cutted = self.cut_indices_in_range(force_rest,
                                                                                       force_max_indices,
                                                                                       force_min_indices,
                                                                                       self.force_max,
                                                                                       self.force_min)

        print("Cycles number= ", len(force_min_indices))
        print("Cycles number after cutting unwanted max-min range= ",
              len(force_min_indices_cutted))

        # TODO I skipped time with presuming it's the first column
        for i in range(1, len(self.columns_headers_list)):
            if self.columns_headers_list[i] != str(self.force_name):
                disp_rest_maxima = disp_rest[force_max_indices_cutted]
                disp_rest_minima = disp_rest[force_min_indices_cutted]
                np.save(os.path.join(self.npy_folder_path, self.file_name + '_' +
                                     self.columns_headers_list[i] + '_max.npy'), disp_rest_maxima)
                np.save(os.path.join(self.npy_folder_path, self.file_name + '_' +
                                     self.columns_headers_list[i] + '_min.npy'), disp_rest_minima)

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
    plot_figure_num = tr.Int(0)

    def _plot_fired(self):
        ax = self.figure.add_subplot()

    def x_plot_fired(self):
        self.plot_figure_num += 1
        plt.draw()
        plt.show()

    data_changed = tr.Event

    def _add_plot_fired(self):

        if False:  # (len(self.plot_list) >= self.plots_num):
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

#        plt.figure(self.plot_figure_num)
        ax = self.figure.add_subplot(1, 1, 1)

        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('kN')
        ax.set_title('Original data', fontsize=20)
        ax.plot(x_axis_array, y_axis_array, 'k', linewidth=0.8)

        self.plot_list.append('{}, {}'.format(x_axis_name, y_axis_name))
        self.data_changed = True
        print('Finished adding plot!')

    def apply_new_subplot(self):
        plt = self.figure
        if (self.plots_num == 1):
            plt.add_subplot(1, 1, 1)
        elif (self.plots_num == 2):
            plot_location = int('12' + str(len(self.plot_list) + 1))
            plt.add_subplot(plot_location)
        elif (self.plots_num == 3):
            plot_location = int('13' + str(len(self.plot_list) + 1))
            plt.add_subplot(plot_location)
        elif (self.plots_num == 4):
            plot_location = int('22' + str(len(self.plot_list) + 1))
            plt.add_subplot(plot_location)
        elif (self.plots_num == 6):
            plot_location = int('23' + str(len(self.plot_list) + 1))
            plt.add_subplot(plot_location)
        elif (self.plots_num == 9):
            plot_location = int('33' + str(len(self.plot_list) + 1))
            plt.add_subplot(plot_location)

    def _add_creep_plot_fired(self):

        plt = self.figure
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

        self.apply_new_subplot()
        plt.xlabel('Cycles number')
        plt.ylabel('mm')
        plt.title('Fatigue creep curve', fontsize=20)
        plt.plot(np.arange(0, disp_max.size), disp_max,
                 'k', linewidth=0.8, color='red')
        plt.plot(np.arange(0, disp_min.size), disp_min,
                 'k', linewidth=0.8, color='green')

        self.plot_list.append('Plot {}'.format(len(self.plot_list) + 1))

        print('Finished adding creep plot!')

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
                    ui.Item('skip_rows'),
                    ui.Item('decimal'),
                    ui.Item('delimiter'),
                    ui.Item('parse_csv_to_npy', show_label=False),
                    label='Filter parameters'
                ),
                ui.VGroup(
                    ui.Item('plots_num'),
                    ui.HGroup(ui.Item('x_axis'), ui.Item('x_axis_multiplier')),
                    ui.HGroup(ui.Item('y_axis'), ui.Item('y_axis_multiplier')),
                    ui.HGroup(ui.Item('add_plot', show_label=False),
                              ui.Item('apply_filters')),
                    ui.HGroup(ui.Item('add_creep_plot', show_label=False)),
                    ui.Item('plot_list'),
                    ui.Item('plot', show_label=False),
                    show_border=True,
                    label='Plotting settings'),
            ),
            ui.VGroup(
                ui.Item('force_name'),
                ui.HGroup(ui.Item('peak_force_before_cycles'),
                          show_border=True, label='Skip noise of ascending branch:'),
                #                     ui.Item('plots_list'),
                ui.VGroup(ui.Item('force_max'),
                          ui.Item('force_min'),
                          show_border=True,
                          label='Cut fake cycles for creep:'),
                ui.Item('generate_filtered_npy', show_label=False),
                show_border=True,
                label='Filters'
            ),
            ui.UItem('figure', editor=MPLFigureEditor(),
                     resizable=True,
                     springy=True,
                     width=0.3,
                     label='2d plots'),
        ),
        title='HCFF Filter',
        resizable=True,
        width=0.6,
        height=0.6

    )


if __name__ == '__main__':
    #     hcff = HCFF(file_csv='C:\\Users\\hspartali\\Desktop\\BeamEnd_Results')
    hcff = HCFF2()
    cut_initial = HCFFilter(name='cut')
    hcff.hcf.add_filter(cut_initial)

    custom_filter = HCFFilter(name='custom')
    cut_initial.add_filter(custom_filter)

    custom_filter.add_filter(HCFFilter(name='average'))
    cut_initial.add_filter(HCFFilter(name='custom differently'))


#    plot_options = [PlotOptions()]
    hcff.configure_traits()
