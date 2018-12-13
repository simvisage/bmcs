#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Dec 21, 2011 by: rch


from etsproxy.traits.api import \
    HasTraits, Instance, Str, Property, cached_property, \
    Enum
from etsproxy.traits.ui.api import \
    View, Item

from fiber_tt_2p import fiber_tt_2p
from fiber_tt_5p import fiber_tt_5p
from fiber_po_8p import fiber_po_8p

import os.path

from os.path import expanduser

HOME_DIR = expanduser("~")
# build directory
BUILD_DIR = os.path.join(HOME_DIR, '.spirrid', 'docs')
# output directory for the documentation
DOCS_DIR = os.path.join('..', 'docs',)
# output directory for the example documentation
EX_OUTPUT_DIR = os.path.join(DOCS_DIR, 'examples')

class GenExampleDoc(HasTraits):

    header = Str('''
Comparison of sampling structure
================================

The different types of sampling for sample size 100. Both variables are randomized with 
normal distribution. 
The exact solution is depicted with the black line. The gray lines indicate the sampling. 
The response diagram correspond to the sampling types (left to right):

Regular grid of random variables
Grid of constant probabilities
Monte Carlo sampling
Latin Hypercube Sampling 
    ''')

    demo_module = fiber_tt_2p

    #===========================================================================
    # Derived traits
    #===========================================================================
    demo_object = Property(depends_on = 'demo_module')
    @cached_property
    def _get_demo_object(self):
        return self.demo_module.create_demo_object()

    qname = Property(depends_on = 'demo_module')
    @cached_property
    def _get_qname(self):
        return self.demo_object.get_qname()

    output_dir = Property(depends_on = 'demo_module')
    @cached_property
    def _get_output_dir(self):
        return os.path.join(EX_OUTPUT_DIR, self.qname)

    rst_file_name = Property(depends_on = 'demo_module')
    @cached_property
    def _get_rst_file_name(self):
        return os.path.join(self.output_dir, 'index.rst')

    def generate_examples_sampling_structure(self):
        dobj = self.demo_object
        dobj.set(fig_output_dir = self.output_dir, show_output = False,
                 dpi = 70,
                 save_output = True, plot_mode = 'figures')
        dobj.sampling_structure()

    def generate_examples_sampling_efficiency(self):
        dobj = self.demo_object
        dobj.set(fig_output_dir = self.output_dir, show_output = False,
                 dpi = 70,
                 save_output = True, plot_mode = 'figures')
        dobj.sampling_efficiency()

    def generate_examples_language_efficiency(self):
        dobj = self.demo_object
        dobj.set(fig_output_dir = self.output_dir, show_output = False,
                 dpi = 70,
                 save_output = True, plot_mode = 'figures')
        dobj.codegen_language_efficiency()

    def generate_examples(self):
        self.generate_examples_sampling_structure()
        self.generate_examples_sampling_efficiency()
        self.generate_examples_language_efficiency()

    def generate_html(self):

        print(('generating documentation for', self.qname, '...'))

        rst_text = '''
================================
Parametric study for %s
================================
        ''' % self.qname

        dobj = self.demo_object

        if dobj.s.q.__doc__ != None:
            rst_text += dobj.s.q.__doc__

        rst_text += self.header

        for st in dobj.sampling_types:
            rst_text += '''
            
.. image:: %s_%s.png
    :width: 24%%

            ''' % (self.qname, st)

        for st in dobj.sampling_types:
            rst_text += '''
                
.. image:: %s_sampling_%s.png
    :width: 24%%
    
            ''' % (self.qname, st)

        rst_text += '\nFollowing spirrid configuration has been used to produce the sampling figures:\n\n'
        rst_text += '\n>>> print demo_object\n' + str(dobj.s) + '\n'

        rst_text += '''
Comparison of execution time for different sampling types
=========================================================
Execution time evaluated for an increasing number of sampling points n_sim:
'''
        for basename in dobj.fnames_sampling_efficiency:
            rst_text += '''
        
.. image:: %s
    :width: 100%%

            ''' % basename
            print(('written file %s', basename))

        rst_text += '\n'

        rst_text += '''
Comparison of efficiency for different code types
=========================================================
Execution time evaluated for an numpy, weave and cython code:
'''
        for basename in dobj.fnames_language_efficiency:
            rst_text += '''
            
.. image:: %s
    :width: 100%%

            ''' % basename
            print(('written file %s', basename))

        rst_text += '\n'

        rst_file = open(self.rst_file_name, 'w')

        rst_file.write(rst_text)

        rst_file.close()

class GenDoc(HasTraits):
    '''
    Configuration of the document generation using sphinx.
    '''
    demo_modules = [fiber_tt_2p, fiber_tt_5p, fiber_po_8p]

    build_mode = Enum('local', 'global')

    build_dir = Property(depends_on = 'build_mode')
    def _get_build_dir(self):
        build_dir = {'local' : '.',
                     'global' : BUILD_DIR }
        return build_dir[self.build_mode]

    html_server = 'root@mordred.imb.rwth-aachen.de:/var/www/docs/spirrid'

    method_dispatcher = {'all' : 'generate_examples',
                         'sampling_structure' : 'generate_examples_sampling_structure',
                         'sampling_efficiency' : 'generate_examples_sampling_efficiency',
                         'language_efficiency' : 'generate_examples_language_efficiency',
                         }

    def generate_examples(self, kind = 'all'):
        method_name = self.method_dispatcher[kind]
        for demo in self.demo_modules:
            ged = GenExampleDoc(demo_module = demo)
            getattr(ged, method_name)()

    def generate_html(self):
        for demo in self.demo_modules:
            ged = GenExampleDoc(demo_module = demo)
            ged.generate_html()

        os.chdir(DOCS_DIR)
        sphings_cmd = 'sphinx-build -b html -E . %s' % self.build_dir
        os.system(sphings_cmd)

    def push_html(self):
        '''
        Push the documentation to the server.
        '''
        rsync_cmd = 'rsync -av --delete %s/ %s' % (self.build_dir, self.html_server)
        os.system(rsync_cmd)

if __name__ == '__main__':

    gd = GenDoc(build_mode = 'global')

    #gd.generate_examples() # kind = 'sampling_efficiency')
    gd.generate_html()
    gd.push_html()
