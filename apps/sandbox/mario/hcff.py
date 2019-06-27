'''
Created on Apr 24, 2019

@author: rch

Changes since Friday June 7 2019 - BMCS coding workshop
    * refresh of MPLFigureEditor - the curve gets reploted
      when the model object changes ist own event - data_changed.
      If the model object has several Matplotlib instances
      it must provide the trait event as its attribute 
      issue the event if the replot should be done
      
      self.data_changed = True

'''
import os

from matplotlib.figure import \
    Figure
import path
from pyface.api import FileDialog
from util.traits.editors import \
    MPLFigureEditor

import numpy as np
import pandas as pd
import traits.api as tr
import traitsui.api as ui

from .reading_csv import read_csv
from .something_traits import Something


class HCFFFilter(tr.HasStrictTraits):

    def process(self):
        raise NotImplementedError


class HCFFFCutNoise(HCFFFilter):

    source_data = tr.Instance(HCFFFilter)

    output_data = tr.Property(depends_on='source_data')

    def _get_output_data(self):
        pass


class HCFFFDetectMinMax(HCFFFilter):

    def process(self):
        pass


class HCFF(tr.HasStrictTraits):
    '''High-Cycle Fatigue Filter
    '''

    #=========================================================================
    # File management
    #=========================================================================
    file_csv = tr.File

    file_basename = tr.Property(depends_on='file_csv')

    @tr.cached_property
    def _get_file_basename(self):
        return os.path.basename(self.file_csv)

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
        self.data = np.loadtxt(self.file_csv,
                               skiprows=self.skip_rows,
                               delimiter=';')
        print(self.data.shape)

    min_max_button = tr.Button()

    def _min_max_button_fired(self):
        f = self.F_input
        f_diff = np.abs(f[1:]) - np.abs(f[:-1])
        g_diff = np.array(f_diff[1:] * f_diff[:-1])
        idx1 = np.array(np.where((g_diff) < 0))
        idx1 = idx1[0] + 1

        F_red = f[idx1]
        U_red = self.U_input[idx1]

        ax = self.figure.add_subplot(223)
        ax.plot(U_red, F_red, 'ro')
        self.data_changed = True


#
#         idx2 = idx1[1:] - idx1[0:-1]
#         idx3 = np.where(np.abs(idx2) == 1)
#         idx1 = list(idx1)

    read_csv_button = tr.Button

    def _read_csv_button_fired(self):
        self.read_csv()
        print('button pressed')

    def read_csv(self):
        '''Read the csv file and transform it to the hdf5 forma.
        The output file has the same name as the input csv file
        with an extension hdf5
        '''
        path = self.file_csv
        basename = path.split('.')
        path2 = ''.join(basename[:-1]) + '.hdf5'

        chunk_size = self.chunk_size
        skip_rows = self.skip_rows

        n_rows = chunk_size - 1 - skip_rows

        f = open(path)
        l = sum(1 for row in f)
        n_chunks = np.int(np.floor(l / chunk_size))
        f = pd.read_csv(path, sep=';', skiprows=skip_rows, nrows=n_rows)
        nf = np.array(f)
        df = pd.DataFrame(nf, columns=['a', 'b', 'c', 'd', 'e', 'f'])

        print('xxx')
        print(df)
        print('xxx')

#         df['a'] = [x.replace(',', '.') for x in df['a']]
#         df['b'] = [x.replace(',', '.') for x in df['b']]
#         df['c'] = [x.replace(',', '.') for x in df['c']]
#         df['d'] = [x.replace(',', '.') for x in df['d']]
#         df['e'] = [x.replace(',', '.') for x in df['e']]
#         df['f'] = [x.replace(',', '.') for x in df['f']]
        # df['g'] = [x.replace(',', '.') for x in df['g']]
        df.to_hdf(path2, 'first', mode='w', format='table')

        for iter_num in range(n_chunks - 1):
            print('i', iter_num)
            f = np.array(pd.read_csv(path, skiprows=(
                iter_num + 1) * chunk_size - 1, nrows=chunk_size, sep=';'))
            nf = np.array(f)
            df = pd.DataFrame(f.astype(str), columns=[
                              'a', 'b', 'c', 'd', 'e', 'f'])
#             df['a'] = [x.replace(',', '.') for x in df['a']]
#             df['b'] = [x.replace(',', '.') for x in df['b']]
#             df['c'] = [x.replace(',', '.') for x in df['c']]
#             df['d'] = [x.replace(',', '.') for x in df['d']]
#             df['e'] = [x.replace(',', '.') for x in df['e']]
#             df['f'] = [x.replace(',', '.') for x in df['f']]
        #     df['g'] = [x.replace(',', '.') for x in df['g']]
            df.to_hdf(path2, 'middle' + np.str(iter_num), append=True)

        f = np.array(pd.read_csv(path, skiprows=n_chunks *
                                 chunk_size - 1, nrows=l - n_chunks * chunk_size, sep=';'))
        nf = np.array(f)
        df = pd.DataFrame(nf, columns=['a', 'b', 'c', 'd', 'e', 'f'])
#         df['a'] = [x.replace(',', '.') for x in df['a']]
#         df['b'] = [x.replace(',', '.') for x in df['b']]
#         df['c'] = [x.replace(',', '.') for x in df['c']]
#         df['d'] = [x.replace(',', '.') for x in df['d']]
#         df['e'] = [x.replace(',', '.') for x in df['e']]
#         df['f'] = [x.replace(',', '.') for x in df['f']]
        # df['g'] = [x.replace(',', '.') for x in df['g']]
        df.to_hdf(path2, 'last', append=True)

    filter_list = tr.List(HCFFFilter)

    figure = tr.Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    clear_button = tr.Button

    def _clear_button_fired(self):
        ax = self.figure.subplot(111)
        ax.clear()

    plot = tr.Button

    def _plot_fired(self):
        self._replot_input()

    F_input = tr.Property(depends_on='file_csv')

    @tr.cached_property
    def _get_F_input(self):
        return self.data[:, 1]

    U_input = tr.Property(depends_on='file_csv')

    @tr.cached_property
    def _get_U_input(self):
        return np.average(self.data[:, (3, 4, 5)], axis=1)

    data_changed = tr.Event

    @tr.on_trait_change('+replot_event')
    def _replot_input(self):
        ax = self.figure.add_subplot(111)
#        ax.plot(self.data[:, self.x_axis], self.data[:, self.y_axis])
        ax.plot(self.U_input, self.F_input)
        self.data_changed = True

    traits_view = ui.View(
        ui.HSplit(
            ui.VSplit(
                ui.HGroup(
                    ui.UItem('open_file_csv'),
                    ui.UItem('file_basename', style='readonly'),
                    label='Input data'
                ),
                ui.Item('clear_button', show_label=False),
                ui.Item('read_loadtxt_button', show_label=False),
                ui.Item('min_max_button', show_label=False),
                ui.VGroup(
                    ui.Item('chunk_size'),
                    ui.Item('skip_rows'),
                    label='Filter parameters'
                ),
                ui.VGroup(
                    ui.Item('read_csv_button', show_label=False),
                    ui.Item('plot', show_label=False)
                )
            ),
            ui.UItem('figure', editor=MPLFigureEditor(),
                     resizable=True,
                     width=900,
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
