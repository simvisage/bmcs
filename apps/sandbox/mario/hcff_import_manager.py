'''
Created on Jun 28, 2019

'''

import os
import string

from pyface.api import FileDialog
import numpy as np
import pandas as pd
import traits.api as tr
import traitsui.api as ui


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
