'''
Created on 25.06.2019

@author: hspartali
'''
import numpy as np
import traits.api as tr
import traitsui.api as ui


class DataTable(tr.HasStrictTraits):
    columns = tr.List(tr.Str)
    data = tr.Array(np.float_)


class HCFFParent(tr.HasStrictTraits):

    filters = tr.List()

    def add_filter(self, child):
        child.source = self
        self.filters.append(child)

    output_table = tr.Property(tr.Dict,
                               depends_on='+inputs')

    def _get_ouput_table(self):
        raise NotImplementedError('Output table not defined')

    columns = tr.Property()

    def _get_columns(self):
        return self.output_table.keys()


class HCFFChild(tr.HasStrictTraits):

    source = tr.WeakRef()


class HCFFilter(HCFFParent, HCFFChild):

    name = tr.Str('Filters name')

    chunk_size = tr.Int(100, input=True)

    output_table = tr.Property(tr.Dict,
                               depends_on='+input')

    @tr.cached_property
    def _get_output_table(self):
        print('accessing data of filter', self.name)
        return {
            'first': self.source.output_table['first'] * 2,
            'second': self.source.output_table['second'] / 2
        }


if __name__ == '__main__':

    f1 = HCFFilter(name='f1')
    f2 = HCFFilter(name='f2')
    f1.add_filter(f2)
