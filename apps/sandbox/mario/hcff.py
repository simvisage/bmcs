'''
Created on Apr 24, 2019

@author: rch
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


class HCFF(tr.HasStrictTraits):
    '''High-Cycle Fatigue Filter
    '''

    something = tr.Instance(Something)

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

    figure = tr.Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    x_axis = tr.Enum(0, 1, 2, 3, 4, 5)
    y_axis = tr.Enum(0, 1, 2, 3, 4, 5)

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
                    label='Filter parameters'
                ),
                ui.VGroup(
                    ui.Item('read_loadtxt_button', show_label=False),
                    ui.Item('read_csv_button', show_label=False),
                    ui.Item('plot', show_label=False)
                )
            ),
            ui.VGroup(
                ui.Item('x_axis'),
                ui.Item('y_axis'),
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
