'''
Created on Jul 9, 2017

@author: rch
'''

from StringIO import StringIO
from os.path import join
from traits.api import \
    HasStrictTraits, Interface, WeakRef, Str, Int, \
    Property, implements, Missing
import numpy as np


class IRItem(Interface):
    '''Interface supporting the protocol or a report item. 
    '''

    def write_report_entry(self):
        pass


class RItem(HasStrictTraits):
    '''Report item.
    '''
    _r = WeakRef

    level = Int
    name = Str

    file_ = Property

    def _get_file_(self):
        return self.reporter.file_


class RInputRecord(RItem):
    '''This class scans the parameters
    that have as metadata one of the input tags and 
    specified in the input tag list.
    '''

    implements(IRItem)

    def _get_rinput_traits(self, **itags):
        rinput_traits = {}
        for itag in itags.keys():
            rinput_traits.update(self.traits(**{itag: True}))
        return rinput_traits

    def _get_rinputs(self, **itags):
        rinput_traits = {}
        for itag in itags.keys():
            rinput_traits.update(self.trait_get(**{itag: True}))
        return rinput_traits

    def write_record_entries(self, f, **itags):
        itraits = self._get_rinput_traits(**itags)
        for name, trait in itraits.items():
            value = getattr(self, name, Missing)
            unit = trait.unit
            symbol = trait.symbol
            label = trait.label
            desc = trait.desc
            f.write('%s & %s & %s & %s & %s & %s \\\\\n' %
                    (name.replace('_', '\_'), str(value),
                     unit, symbol, label, desc))

    def write_record(self, f, rdir, **itags):
        tag_io = StringIO()
        self.write_record_entries(tag_io, **itags)
        self.write_figure(tag_io, join(rdir, 'fig' + str(id(self)) + '.pdf'))
        f.write(tag_io.getvalue())

    def write_figure(self, f, fname):
        pass


class RInputSection(RInputRecord):

    implements(IRItem)

    def get_records(self):
        records = {}
        for name, value in self.trait_get(report=True).items():
            records[name] = value
        return records

    def write_record(self, f, rdir, **itags):
        tag_io = StringIO()
        self.write_record_entries(tag_io, **itags)
        f.write(tag_io.getvalue())
        records = self.get_records()
        for name, record in records.items():
            tag_io = StringIO()
            clname = record.__class__.__name__
            record.write_record(tag_io, rdir, **itags)
            tag_string = tag_io.getvalue()
            if len(tag_string) > 0:
                f.write(r'''\hline
\multicolumn{6}{l}{%s : %s}\\ \hline

''' % (name.replace('_', '\_'), clname))
                f.write(tag_string)

    def write_report(self, f, rdir, **itags):
        f.write(r'''\section*{Input}\begin{tabular}{lrrclL{4cm}}\hline
Name & Value & Unit & Symbol & Label & Description \\\hline \hline
''')
        self.write_record(f, rdir, **itags)
        f.write(r'''\hline \end{tabular}

''')


class ROutputRecord(RItem):

    def write_record(self, f, rdir, **itags):
        tag_io = StringIO()
        self.write_figure(tag_io, join(rdir, 'fig' + str(id(self)) + '.pdf'))
        f.write(tag_io.getvalue())


class ROutputSection(RInputSection):

    implements(IRItem)

    def get_records(self):
        raise NotImplemented

    def write_record(self, f, rdir, **itags):
        records = self.get_records()
        record_fig_list = []
        for record in records:
            tag_io = StringIO()
            clname = record.__class__.__name__
            record.write_record(tag_io, rdir, **itags)
            tag_string = tag_io.getvalue()
            if len(tag_string) > 0:
                record_fig_list.append(tag_string)
        record_arr = np.array(record_fig_list).reshape(-1, 2)
        for r1, r2 in record_arr:
            f.write(r'''
\noindent
\begin{tabular}{L{7.5cm}L{7.5cm}}''')
            f.write(r'%s & %s \\' % (r1, r2))
            f.write(r'''\end{tabular}
''')

    def write_report(self, f, rdir, **itags):
        f.write(r'''\section*{Output}''')
        self.write_record(f, rdir, **itags)
