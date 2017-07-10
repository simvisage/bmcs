'''
Created on Jul 9, 2017

@author: rch
'''

from StringIO import StringIO

from traits.api import \
    HasStrictTraits, Interface, WeakRef, Str, Int, \
    Property, implements, Missing


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

    def _get_rinput_traits(self, ITAG):
        kw = {ITAG: True}
        return self.traits(**kw)

    def _get_rinputs(self, ITAG):
        kw = {ITAG: True}
        return self.trait_get(**kw)

    def write_report_entry(self, ITAG, f):
        for name, trait in self._get_rinput_traits(ITAG).items():
            value = getattr(self, name, Missing)
            unit = trait.unit
            symbol = trait.symbol
            label = trait.label
            f.write('%s & %s & %s & %s & %s \\\\\n' %
                    (name.replace('_', '\_'), str(value),
                     unit, symbol, label))


class RInputSubDomain(RInputRecord):

    implements(IRItem)

    def write_report(self, f):
        f.write(r'''\begin{tabular}{|l|r|r|c|l|}\hline
Code & Value & Unit & Symbol & Label \\
''')
        for ITAG in self._r.INPUTTAGS:
            tag_io = StringIO()
            for name, ritem in self._get_rinputs(ITAG).items():
                print 'name', name
                ritem.write_report_entry(ITAG, tag_io)
            tag_string = tag_io.getvalue()
            if len(tag_string) > 0:
                f.write(r'''\hline
\multicolumn{5}{|c|}{%s}\\ \hline
''' % self._r.TAGLABELS[ITAG])
                f.write(tag_string)
        f.write(r'''\hline \end{tabular}''')
