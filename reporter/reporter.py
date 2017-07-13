'''
Created on Jul 9, 2017

@author: rch
'''

from StringIO import StringIO
import os
import subprocess
import sys
import tempfile

from traits.api import \
    HasStrictTraits, Str, List, Property, \
    cached_property, Dict

from report_item import IRItem


class Reporter(HasStrictTraits):

    INPUTTAGS = List(['MAT', 'GEO', 'CS', 'ALG', 'BC', 'MESH'])

    itags = Property(depends_on='INPUTTAGS')

    @cached_property
    def _get_itags(self):
        return {tag: True for tag in self.INPUTTAGS}

    report_items = List(IRItem)

    report_name = Str('unnamed')

    bmcs_example_dir = Str('examples')

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

    rsubfile_tex = Property(depends_on='report_name')

    @cached_property
    def _get_rsubfile_tex(self):
        return os.path.join(self.rdir, 'rs_' + self.report_name + '.tex')

    rfile_pdf = Property(depends_on='report_name')

    @cached_property
    def _get_rfile_pdf(self):
        return os.path.join(self.rdir, 'r_' + self.report_name + '.pdf')

    title = Property

    def _get_title(self):
        return self.ritems[0].title

    desc = Property

    def _get_desc(self):
        return self.ritems[0].desc

    def write(self):

        preamble = r'''\documentclass{article}
\oddsidemargin=0cm
\topmargin=-1cm
\textwidth=16cm
\textheight=25cm
\usepackage{graphicx}          % include graphics
\usepackage{array}
\newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{R}[1]{>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}


\usepackage[most]{tcolorbox}
\newcounter{bmcsexample}
\usepackage{xparse}
\usepackage{lipsum}

\def\exampletext{Example} % If English

\NewDocumentEnvironment{bmcsexample}{ O{} }
{
\colorlet{colexam}{red!55!black} % Global example color
\newtcolorbox[use counter=bmcsexample]{bmcsexamplebox}{%
    % Example Frame Start
    empty,% Empty previously set parameters
    title={\exampletext: #1},% use \thetcbcounter to access the bmcsexample counter text
    % Attaching a box requires an overlay
    attach boxed title to top left,
       % Ensures proper line breaking in longer titles
       minipage boxed title,
    % (boxed title style requires an overlay)
    boxed title style={empty,size=minimal,toprule=0pt,top=4pt,left=3mm,overlay={}},
    coltitle=colexam,fonttitle=\bfseries,
    before=\par\medskip\noindent,parbox=false,boxsep=0pt,left=3mm,right=0mm,top=2pt,breakable,pad at break=0mm,
       before upper=\csname @totalleftmargin\endcsname0pt, % Use instead of parbox=true. This ensures parskip is inherited by box.
    % Handles box when it exists on one page only
    overlay unbroken={\draw[colexam,line width=.5pt] ([xshift=-0pt]title.north west) -- ([xshift=-0pt]frame.south west); },
    % Handles multipage box: first page
    overlay first={\draw[colexam,line width=.5pt] ([xshift=-0pt]title.north west) -- ([xshift=-0pt]frame.south west); },
    % Handles multipage box: middle page
    overlay middle={\draw[colexam,line width=.5pt] ([xshift=-0pt]frame.north west) -- ([xshift=-0pt]frame.south west); },
    % Handles multipage box: last page
    overlay last={\draw[colexam,line width=.5pt] ([xshift=-0pt]frame.north west) -- ([xshift=-0pt]frame.south west); },%
    }
\begin{bmcsexamplebox}}
{\end{bmcsexamplebox}\endlist}


\begin{document}
        '''
        postamble = r'''
\end{document}
        '''
        preamble_subfile = r'''\documentclass[main.tex]{subfiles}
\begin{document}
'''

        rfile_io = StringIO()

        rfile_io.write(r'''
\begin{bmcsexample}[%s]
\noindent %s \\[3mm]
\begin{center}
''' % (self.title, self.desc))
        self.ritems[0].write_report(rfile_io, self.rdir, **self.itags)
        rfile_io.write(r'''
\end{center}
''')
        self.ritems[1].write_report(rfile_io, self.rdir, **self.itags)
        rfile_io.write(r'''
\end{bmcsexample}
''')

        rfile_str = rfile_io.getvalue()
        full_path = os.path.join('{examples', self.report_name, 'fig')
        rsubfile_str = rfile_str.replace('{fig', full_path)

        with open(self.rfile_tex, "w") as f:
            f.write(preamble)
            f.write(rfile_str)
            f.write(postamble)

        with open(self.rsubfile_tex, "w") as f:
            f.write(preamble_subfile)
            f.write(rsubfile_str)
            f.write(postamble)

    def show_tex(self):
        with open(self.rfile_tex, 'r') as f:
            texsource = f.read()
            print texsource

    def show_stex(self):
        with open(self.rfile_subs_tex, 'r') as f:
            texsource = f.read()
            print texsource

    def run_pdflatex(self):
        cmd = ['pdflatex', '-interaction', 'nonstopmode', self.rfile_tex]
        proc = subprocess.Popen(cmd, cwd=self.rdir)
        proc.communicate()

        retcode = proc.returncode
        if not retcode == 0:
            os.unlink(self.rfile_tex)
            raise ValueError('Error {} executing command: {}'.format(
                retcode, ' '.join(cmd)))

    def show_pdf(self):
        if sys.platform == 'linux2':
            subprocess.call(["xdg-open", self.rfile_pdf])
        else:
            os.startfile(self.rfile_pdf)
