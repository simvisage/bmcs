'''
Created on 25.06.2019

@author: hspartali
'''
import os

from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

import numpy as np
import traits.api as tr
import traitsui.api as ui

from .hcff_filter import HCFFilter


class SmoothAscendingBranchFilter(HCFFilter):

    peak_force_before_cycles = tr.Float
    columns_headers_list = tr.List()
    npy_folder_path = tr.Str
    file_name = tr.Str

    def apply(self):

        force = np.load(
            os.path.join(self.npy_folder_path,
                         self.file_name + '_' +
                         self.force_name + '.npy')
        ).flatten()
        peak_force_before_cycles_index = np.where(
            abs((force)) > abs(self.peak_force_before_cycles))[0][0]

        for i in range(1, len(self.columns_headers_list)):
            disp = np.load(
                os.path.join(self.npy_folder_path,
                             self.file_name + '_' +
                             self.columns_headers_list[i] + '.npy')).flatten()

            disp_ascending = disp[0:peak_force_before_cycles_index]
            disp_rest = disp[peak_force_before_cycles_index:]
            disp_ascending = savgol_filter(
                disp_ascending, window_length=51, polyorder=2)
            filtered_disp = np.concatenate((disp_ascending, disp_rest))

            np.save(
                os.path.join(self.npy_folder_path, self.file_name + '_'
                             + self.columns_headers_list[i] +
                             '_filtered.npy'), filtered_disp)
