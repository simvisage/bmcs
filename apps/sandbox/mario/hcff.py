'''
Created on Apr 24, 2019

@author: rch
'''
from apps.sandbox.mario import reading_csv
import traits.api as tr

from .reading_csv import read_csv


class HCFF(tr.HasStrictTraits):

    file = tr.File

    chunk_size = tr.Int(10000, auto_set=False, enter_set=True)

    @tr.on_trait_change('chunk_size')
    def chunk_size_changed(self):
        print('chunk-size changed')

    read_csv_button = tr.Button

    def _read_csv_button_fired(self):
        read_csv(name=self.file1)


if __name__ == '__main__':
    hcff = HCFF()

    hcff.configure_traits()
