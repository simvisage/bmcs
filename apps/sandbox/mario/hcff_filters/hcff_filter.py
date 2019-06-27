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


class HCFFChild(tr.HasStrictTraits):

    source = tr.WeakRef()


class HCFFilter(HCFFParent, HCFFChild):

    name = tr.Str('Filters name')

#     output_table = tr.Property(tr.Instance(DataTable),
#                                depends_on='+inputs')


if __name__ == '__main__':

    f1 = HCFFilter(name='f1')
    f1.add_filter(HCFFilter(name='f2'))
    f1.configure_traits()
