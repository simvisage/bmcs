'''
Created on Jul 9, 2017

@author: rch
'''

from io import StringIO
from traits.api import \
    HasStrictTraits, Interface, Str, \
    Dict, Property, provides, Missing
import numpy as np


class IRItem(Interface):
    '''Interface supporting the protocol or a report item. 
    '''

    def write_report_entry(self):
        pass


class RItem(HasStrictTraits):
    '''Report item.
    '''


@provides(IRItem)
class RInputRecord(RItem):
    '''This class scans the parameters
    that have as metadata one of the input tags and 
    specified in the input tag list.
    '''

    name = Str

    itags = Dict

    def get_subrecords(self):
        subrecords = {}
        for name, value in list(self.trait_get(report=True).items()):
            subrecords[name] = value
        return subrecords

    def get_subrecord_traits(self):
        return self.traits(report=True)

    def _get_rinput_traits(self, itags=None):
        if itags == None:
            itags = self.itags
        input_traits = {}
        for itag in list(itags.keys()):
            input_traits.update(self.traits(**{itag: True}))
        return input_traits

    def yield_rinput_traits(self, path='', itags=None):
        return {path + '.' + name: trait
                for name, trait in list(self._get_rinput_traits(itags).items())
                }

    def _get_rinputs(self, itags=None):
        if itags == None:
            itags = self.itags
        rinput_traits = {}
        for itag in list(itags.keys()):
            rinput_traits.update(self.trait_get(**{itag: True}))
        return rinput_traits

    def _repr_latex_(self):
        io = StringIO()
        io.write(r'''
        \begin{array}{lrrl}\hline
        ''')
        itraits = self._get_rinput_traits()
        for name, trait in list(itraits.items()):
            value = getattr(self, name, Missing)
            unit = trait.unit
            symbol = trait.symbol
            label = trait.label
            desc = trait.desc
            io.write(r'''\textrm{%s} & %s = %s & \textrm{[%s]} & \textrm{%s}  \\
            ''' % (name, symbol, str(value), unit, desc))
        io.write(r'''\hline
        ''')
        records = self.get_subrecord_traits()
        for name, trait in list(records.items()):
            value = getattr(self, name, Missing)
            clname = value.__class__.__name__
            desc = trait.desc
            io.write(r'''\textbf{%s} & \textrm{%s} & & \textrm{%s} \\
            ''' % (name, clname, desc))
        io.write(r'''\hline
        \end{array}
        ''')
        return io.getvalue()

    def write_tex_table_record_entries(self, f, path='', itags=None):
        itraits = self._get_rinput_traits(itags)
        for name, trait in list(itraits.items()):
            value = getattr(self, name, Missing)
            unit = trait.unit
            symbol = trait.symbol
            label = trait.label
            desc = trait.desc
            f.write(r'''\texttt{%s%s} & $%s$ = %s [%s] & {\footnotesize %s}  \\
            ''' %
                    (path, name.replace('_', '\_'), symbol, str(value), unit, desc))

    def write_tex_table_record(self, f, path='', rdir=None,
                               rel_study_path=None,
                               itags=None):
        tag_io = StringIO()
        self.write_tex_table_record_entries_with_figs(
            tag_io, path, rdir, rel_study_path, itags)
        f.write(tag_io.getvalue())
        records = self.get_subrecords()
        for name, record in list(records.items()):
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

    def write_tex_table_record_entries_with_figs(self, f, path='', rdir=None,
                                                 rel_study_path=None,
                                                 itags=None):
        entries_io = StringIO()
        figures_io = StringIO()
        self.write_tex_table_record_entries(entries_io, path, itags)
        if rdir != None:
            self.write_figure(figures_io, rdir, rel_study_path)
        entries_str = entries_io.getvalue()
        figures_str = figures_io.getvalue()
        f.write(entries_str)
        f.write(figures_str)

    def write_figure(self, f, rdir=None, rel_study_path=None):
        pass

    inputs = Property()

    def _get_inputs(self):
        inputs = StringIO()
        self.write_tex_table(inputs)
        return inputs.getvalue()

    def write_tex_table(self, f, rdir=None, rel_study_path=None, itags=None):
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

    def yield_rinput_section(self, path='', itags=None):
        rinput_traits = self.yield_rinput_traits(path, itags)
        for name, trait in list(self.get_subrecords().items()):
            rinput_traits.update(
                trait.yield_rinput_section(path + name + '.', itags))
        return rinput_traits


class ROutputItem(RItem):

    def write_record(self, f, rdir, rel_study_path=None, itags=None):
        tag_io = StringIO()
        self.write_figure(tag_io, rdir, rel_study_path)
        f.write(tag_io.getvalue())


@provides(IRItem)
class ROutputSection(RItem):

    def write_records(self, f, rdir=None, rel_study_path=None, itags=None):
        records = self.get_subrecords()
        record_fig_list = []
        for record in records:
            tag_io = StringIO()
            record.write_record(tag_io, rdir, rel_study_path, itags)
            tag_string = tag_io.getvalue()
            if len(tag_string) > 0:
                record_fig_list.append(tag_string)
        record_arr = np.array(record_fig_list).reshape(-1, 2)
        for r1, r2 in record_arr:
            f.write(r'''
\noindent
\begin{longtable}{L{7.5cm}L{7.5cm}}''')
            f.write(r'%s & %s \\' % (r1, r2))
            f.write(r'''\end{longtable}
''')

    def write_report(self, f='', rdir=None, itags=None):
        self.write_record(f, rdir, itags)
