'''
Created on Jul 9, 2017

@author: rch
'''

import os
import subprocess
import sys
import tempfile

from traits.api import \
    HasStrictTraits, Str, List, Property, \
    cached_property, Dict

from report_item import IRItem


class Reporter(HasStrictTraits):

    INPUTTAGS = List(['MAT', 'GEO', 'CS', 'ALG'])

    TAGLABELS = Dict(dict(MAT='Material',
                          GEO='Geometry',
                          CS='Cross-section',
                          ALG='Algorithm'))

    report_items = List(IRItem)

    report_name = Str('<unnamed>')

    ritems = Property

    def _get_ritems(self):
        for ritem in self.report_items:
            ritem._r = self
        return self.report_items

    rdir = Property(depends_on='report_name')

    @cached_property
    def _get_rdir(self):
        tdir = tempfile.mkdtemp()
        rdir = os.path.join(tdir, self.report_name)
        os.mkdir(rdir)
        return rdir

    rfile_tex = Property(depends_on='report_name')

    @cached_property
    def _get_rfile_tex(self):
        return os.path.join(self.rdir, 'r_' + self.report_name + '.tex')

    rfile_pdf = Property(depends_on='report_name')

    @cached_property
    def _get_rfile_pdf(self):
        return os.path.join(self.rdir, 'r_' + self.report_name + '.pdf')

    def write(self):

        preamble = r'''\documentclass{article}
\begin{document}
        '''
        postamble = r'''
\end{document}
        '''

        f = open(self.rfile_tex, 'w')
        f.write(preamble)

        for ritem in self.ritems:
            ritem.write_report(f)

        f.write(postamble)

        f.close()

    def show_tex(self):
        with open(self.rfile_tex, 'r') as f:
            texsource = f.read()
            print 'texsource\n', texsource

    def run_pdflatex(self):
        cmd = ['pdflatex', '-interaction', 'nonstopmode', self.rfile_tex]
        proc = subprocess.Popen(cmd, cwd=self.rdir)
        proc.communicate()

        retcode = proc.returncode
        if not retcode == 0:
            os.unlink(self.rfile_tex)
            raise ValueError('Error {} executing command: {}'.format(
                retcode, ' '.join(cmd)))

        os.unlink(self.rfile_tex)

    def show_pdf(self):
        if sys.platform == 'linux2':
            subprocess.call(["xdg-open", self.rfile_pdf])
        else:
            pass