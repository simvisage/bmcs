'''
Created on 25.06.2019

@author: hspartali
'''
import os

from traitsui.api import InstanceEditor

import numpy as np
import traits.api as tr
import traitsui.api as ui

from .plot_settings import PlotSettings


class DataTable(tr.HasStrictTraits):
    columns = tr.List(tr.Str)
    data = tr.Array(np.float_)


class HCFFParent(tr.HasStrictTraits):

    filters = tr.List()

    def add_filter(self, child):
        child.source = self
        self.filters.append(child)

    output_table = tr.Property(tr.Dict, depends_on='+inputs')

    def _get_ouput_table(self):
        raise NotImplementedError('Output table not defined')

    columns = tr.Property()

    def _get_columns(self):
        return self.output_table.keys()


class HCFFChild(tr.HasStrictTraits):

    source = tr.WeakRef()


class HCFFilter(HCFFParent, HCFFChild):

    name = tr.Str('Filters name')
    check = tr.Str('homam_check')

    chunk_size = tr.Int(100, input=True)

    output_table = tr.Property(tr.Dict, depends_on='+input')

    def get_name(self):
        return 'homam'


class CutAscendingPartNoiseFilter(HCFFilter):

    name = tr.Str('Cut ascending part noise')

    output_table = tr.Property(tr.Dict, depends_on='+input')

    force_name = tr.Str('Kraft', input=True)
    peak_force_before_cycles = tr.Float(30, input=True)

    @tr.cached_property
    def _get_output_table(self):
        print('accessing data of filter', self.name)
        print('--------------------------')
        print(super(CutAscendingPartNoiseFilter, self).get_name())
        print(super(CutAscendingPartNoiseFilter, self).check)
        print('--------------------------')
        # 1- Export filtered force
        force = np.load(os.path.join(self.source.import_manager.npy_folder_path,
                                     self.source.import_manager.file_name + '_'
                                     + self.force_name + '.npy')).flatten()
        peak_force_before_cycles_index = np.where(
            abs((force)) > abs(self.peak_force_before_cycles))[0][0]
        force_rest = force[peak_force_before_cycles_index:]

        force_filtered = np.concatenate((np.zeros((1)), force_rest))
        np.save(os.path.join(self.source.import_manager.npy_folder_path, self.source.import_manager.file_name + '_'
                             + self.force_name
                             + '_CAPN.npy'), force_filtered)

        # 2- Export filtered displacements
        # TODO I skipped time with presuming it's the first column
        for i in range(1, len(self.source.import_manager.columns_headers_list)):
            if self.source.import_manager.columns_headers_list[i] != str(self.force_name):

                disp = np.load(os.path.join(self.source.import_manager.npy_folder_path, self.source.import_manager.file_name
                                            + '_' +
                                            self.source.import_manager.columns_headers_list[i]
                                            + '.npy')).flatten()
                disp_rest = disp[peak_force_before_cycles_index:]
                disp_filtered = np.concatenate((np.zeros((1)), disp_rest))
                np.save(os.path.join(self.source.import_manager.npy_folder_path, self.source.import_manager.file_name + '_'
                                     + self.source.import_manager.columns_headers_list[i]
                                     + '_CAPN.npy'), disp_filtered)

        print('Filtered npy files are generated.')

        return {
            'first': self.source.output_table['first'] * 2,
            'second': self.source.output_table['second'] / 2
        }

    view = ui.View('name', 'force_name', 'peak_force_before_cycles')
