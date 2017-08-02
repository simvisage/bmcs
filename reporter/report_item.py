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


class RInputRecord(RItem):
    '''This class scans the parameters
    that have as metadata one of the input tags and 
    specified in the input tag list.
    '''

    implements(IRItem)

    name = Str

    def get_subrecords(self):
        subrecords = {}
        for name, value in self.trait_get(report=True).items():
            subrecords[name] = value
        return subrecords

    def _get_rinput_traits(self, itags):
        input_traits = {}
        for itag in itags.keys():
            input_traits.update(self.traits(**{itag: True}))
        return input_traits

    def yield_rinput_traits(self, path, itags):
        return {path + '.' + name: trait
                for name, trait in self._get_rinput_traits(itags).items()
                }

    def _get_rinputs(self, itags):
        rinput_traits = {}
        for itag in itags.keys():
            rinput_traits.update(self.trait_get(**{itag: True}))
        return rinput_traits

    def write_tex_table_record_entries(self, f, path, itags):
        itraits = self._get_rinput_traits(itags)
        for name, trait in itraits.items():
            value = getattr(self, name, Missing)
            unit = trait.unit
            symbol = trait.symbol
            label = trait.label
            desc = trait.desc
            f.write(r'''\texttt{%s%s} & %s = %s [%s] & {\footnotesize %s}  \\
            ''' %
                    (path, name.replace('_', '\_'), symbol, str(value), unit, desc))

    def write_tex_table_record(self, f, path, rdir, rel_study_path, itags):
        tag_io = StringIO()
        self.write_tex_table_record_entries_with_figs(
            tag_io, path, rdir, rel_study_path, itags)
        f.write(tag_io.getvalue())
        records = self.get_subrecords()
        for name, record in records.items():
            tag_io = StringIO()
            clname = record.__class__.__name__
            nm = name.replace('_', '\_')
            record.write_tex_table_record(
                tag_io, path + nm + '.', rdir, rel_study_path, itags)
            tag_string = tag_io.getvalue()
            if len(tag_string) > 0:
                f.write(r'''\midrule
\multicolumn{3}{l}{\textbf{\textsf{%s: %s}}}\\

''' % (clname, name.replace('_', '\_')))
                f.write(tag_string)

    def write_tex_table_record_entries_with_figs(self, f, path, rdir,
                                                 rel_study_path, itags):
        entries_io = StringIO()
        figures_io = StringIO()
        self.write_tex_table_record_entries(entries_io, path, itags)
        self.write_figure(figures_io, rdir, rel_study_path)
        entries_str = entries_io.getvalue()
        figures_str = figures_io.getvalue()
        f.write(entries_str)
        f.write(figures_str)

    def write_figure(self, f, rdir, rel_study_path):
        pass

    def write_tex_table(self, f, rdir, rel_study_path, itags):
        f.write(r'''
{\scriptsize 
\begin{longtable}{lrp{4cm}}\toprule
\textbf{\textsf{Model parameter}} 
& 
\textbf{\textsf{Symbol = Value [Unit]}} 
&
\textbf{\textsf{Description}}  \\\midrule \midrule
''')
        self.write_tex_table_record(f, '', rdir, rel_study_path, itags)
        f.write(r'''\bottomrule 
\end{longtable}
}
''')

    def yield_rinput_section(self, path, itags):
        rinput_traits = self.yield_rinput_traits(path, itags)
        for name, trait in self.get_subrecords().items():
            rinput_traits.update(
                trait.yield_rinput_section(path + name + '.', itags))
        return rinput_traits


class ROutputItem(RItem):

    def write_record(self, f, rdir, rel_study_path, itags):
        tag_io = StringIO()
        self.write_figure(tag_io, rdir, rel_study_path)
        f.write(tag_io.getvalue())


class ROutputSection(RItem):

    implements(IRItem)

    def write_records(self, f, rdir, rel_study_path, itags):
        records = self.get_subrecords()
        record_fig_list = []
        for record in records:
            tag_io = StringIO()
            clname = record.__class__.__name__
            record.write_record(tag_io, rdir, rel_study_path, itags)
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

    def write_report(self, f, rdir, itags):
        self.write_record(f, rdir, itags)
