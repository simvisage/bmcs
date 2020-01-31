'''
Created on Jul 9, 2017

@author: rch

Using the reporter
==================
Reporter uses the report items

For single calculation
----------------------
put the inputs and outputs as separate report items

For parametric studies
----------------------
put the input as the input report item
how to mark the parameters that are varied?
can it be done by setting the metadata dynamically with value range?
then the reporter might specify the range of values included.

The p-study driver might gather the parameter values from
the traits and define a schedule of the study. 

A cumulative viz_adapter might be used to gather the results
of the simulation in the desired form.

In other 
An extra list of
viz_adapters might be defined to plot the values from 
the calculation additively.
'''

from io import StringIO
import os
from shutil import copyfile
import subprocess
import sys
import tempfile

from traits.api import \
    HasStrictTraits, Str, List, Property, \
    cached_property, Instance, Constant

from .report_item import RInputRecord, ROutputSection


class ReportStudy(HasStrictTraits):

    name = Str

    title = Str('<no title>')
    desc = Str('''<no description>''')

    input = Instance(RInputRecord)

    output = Instance(ROutputSection)

    preamble_subfile = Constant(r'''\documentclass[main.tex]{subfiles}
\begin{document}
''')

    postable_subfile = Constant(r'''\end{document}
    ''')

    def write_subfile_tex(self, main_file, examples_dir, itags):
        study_name = self.name
        rel_study_path = os.path.join('examples', study_name)
        rdir = os.path.join(examples_dir, study_name)
        if not os.path.exists(rdir):
            os.mkdir(rdir)
        subfile_tex_name = study_name + '.tex'
        rfile_tex = os.path.join(rdir, subfile_tex_name)

        with open(rfile_tex, 'w') as subfile:
            subfile.write(self.preamble_subfile)
            subfile.write(r'''\begin{bmcsex}{%s}{%s}
\noindent %s \\
\begin{center}
            ''' % (self.title, study_name, self.desc))
            self.write_tex_input(subfile, rdir,
                                 rel_study_path, itags)
            self.write_tex_output(subfile, rdir,
                                  rel_study_path, itags)
            subfile.write(r'''\end{center}
            ''')
            subfile.write(r'''\end{bmcsex}
''')
            subfile.write(self.postable_subfile)

        main_file.write(r'''\subfile{%s}
        ''' % os.path.join(rdir, study_name))

    name = Property

    def _get_name(self):
        return self.input.name

    def write_tex_input(self, subfile, rdir,
                        rel_study_path, itags):
        self.input.write_tex_table(subfile, rdir,
                                   rel_study_path, itags)

    def write_tex_output(self, subfile, rdir,
                         rel_study_path, itags):
        if self.output:
            self.output.write_records(subfile, rdir,
                                      rel_study_path, itags)

    def yield_rinput_traits(self):
        model = self.input
        return model.yield_rinput_section('', self.itags)


class Reporter(HasStrictTraits):

    INPUTTAGS = List(['MAT', 'GEO', 'CS', 'ALG', 'BC', 'MESH'])

    itags = Property(depends_on='INPUTTAGS')

    @cached_property
    def _get_itags(self):
        return {tag: True for tag in self.INPUTTAGS}

    studies = List(ReportStudy)

    preamble = Constant(r'''
% !TeX document-id = {c563f3ab-d450-45cf-a935-b20492a6f8a6}
% !TeX program = pdflatex
% !BIB program = biber
% !TeX encoding = UTF-8
% !TeX spellcheck = en_GB
% Requires:
% texlive 2016: Ubuntu 14.04 and 16.04 PPA: ppa:jonathonf/texlive
%  pdflatex --enable-write18, biber, makeglossary
%  perl, ghostscript
%
\documentclass{lib/scidoc}
%
\input{lib/packages}
\input{lib/example}
\input{lib/bmcsex}
%\input{shortcuts/defs}
%\input{bib/bibliographies}

\title{Brittle-Matrix Composite Structures}
\author{Rostislav Chudoba}
%
\begin{document}
%
%\maketitle

\pagenumbering{arabic}
%
% TABLE OF CONTENTS
%\microtypesetup{protrusion=false}  % disable protrusion
%\pdfbookmark[0]{Contents}{toc}
%\tableofcontents % prints Table of Contents
% LISTS AND GLOSSARIES
%\listoffigures
%  \clearpage
%  \listoftables
%  \clearpage
%  \printglossary[title = Nomenclature,  %
%           type = main,    %
%          ]
%  \glsresetall
%  \clearpage
\microtypesetup{protrusion=true}  % enables protrusion
\clearpage
% 
% CONTENT
%
%
''')

    postamble = Constant(r'''
\end{document}
        ''')

    report_name = Str('example_report')

    report_dir = Property()

    @cached_property
    def _get_report_dir(self):
        tmpdir = tempfile.mkdtemp()
        report_dir = os.path.join(tmpdir, self.report_name)
        os.mkdir(report_dir)
        return report_dir

    example_dirname = Str('examples')

    example_dir = Property

    @cached_property
    def _get_example_dir(self):
        exa_dir = os.path.join(self.report_dir, self.example_dirname)
        os.mkdir(exa_dir)
        return exa_dir

    tex_file = Property(depends_on='report_name')

    @cached_property
    def _get_tex_file(self):
        return os.path.join(self.report_dir, self.report_name + '.tex')

    pdf_file = Property(depends_on='report_name')

    @cached_property
    def _get_pdf_file(self):
        return os.path.join(self.report_dir, self.report_name + '.pdf')

    def copy_lib_files(self):
        this_dir, this_filename = os.path.split(__file__)
        tex_source_lib = os.path.join(this_dir, 'texfiles')
#        tex_source_lib = os.path.join(os.sep, 'home', 'rch', 'bmcs_preamble')
        target_dir = os.path.join(self.report_dir, 'lib')
        os.mkdir(target_dir)
        files = ['packages.tex', 'bmcsex.tex', 'example.tex', 'scidoc.cls']
        for f in files:
            copyfile(os.path.join(tex_source_lib, f),
                     os.path.join(target_dir, f))

    def write(self):
        self.copy_lib_files()
        rfile_io = StringIO()
        for study in self.studies:
            study.write_subfile_tex(rfile_io, self.example_dir, self.itags)
            rfile_str = rfile_io.getvalue()

        with open(self.tex_file, "w") as f:
            f.write(self.preamble)
            f.write(rfile_str)
            f.write(self.postamble)

    def show_tex(self):
        with open(self.tex_file, 'r') as f:
            texsource = f.read()
            print(texsource)

    def run_pdflatex(self):
        cmd = ['pdflatex', '-interaction', 'nonstopmode', self.tex_file]
        proc = subprocess.Popen(cmd, cwd=self.report_dir)
        proc.communicate()

        retcode = proc.returncode
        if not retcode == 0:
            raise ValueError('Error {} executing command: {}'.format(
                retcode, ' '.join(cmd)))

    def show_pdf(self):
        if sys.platform == 'linux2':
            subprocess.call(["xdg-open", self.pdf_file])
        else:
            os.startfile(self.pdf_file)
