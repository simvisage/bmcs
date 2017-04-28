#!/usr/bin/env python
'''
Created on Dec 17, 2016

Toplevel script to start oricreate 

@author: rch
'''

import sys


# A oricreate mayavi instance so we can close it correctly.
oricreate = None


def main():
    """This starts up the oricreate application.
    """
    global mxn

    # Make sure '.' is in sys.path
    if '' not in sys.path:
        sys.path.insert(0, '')
    # Start the app.
    from traits.etsconfig.api import ETSConfig
    # Check that we have a traits backend installed
    from traitsui.toolkit import toolkit
    toolkit()  # This forces the selection of a toolkit.
    if ETSConfig.toolkit in ('null', ''):
        raise ImportError('''Could not import backend for traits
________________________________________________________________________________
Make sure that you have either the TraitsBackendWx or the TraitsBackendQt
projects installed. If you installed Oricreate with easy_install, try easy_install
<pkg_name>. easy_install Oricreate[app] will also work.
If you performed a source checkout, be sure to run 'python setup.py install'
in Traits, TraitsGUI, and the Traits backend of your choice.
Also make sure that either wxPython or PyQT is installed.
wxPython: http://www.wxpython.org/
PyQT: http://www.riverbankcomputing.co.uk/software/pyqt/intro
'''
                          )

    from mxn.view import \
        MxNTreeView

    from mxn.use_cases import \
        UseCaseContainer, UCParametricStudy

    mxn_ps = UCParametricStudy()
    mxn_ps.element_to_add = 'mxndiagram'
    mxn_ps.add_element = True
    mxn_ps.tree_node_list[-1].content.cs.matrix_cs.geo.height = 0.06
    mxn_ps.tree_node_list[-1].node_name = 'Study #1 - height 6 cm'
    mxn_ps.add_element = True
    mxn_ps.tree_node_list[-1].linestyle = 'dashed'
    mxn_ps.tree_node_list[-1].content.cs.matrix_cs.geo.height = 0.07
    mxn_ps.tree_node_list[-1].node_name = 'Study #2 - height 7 cm'

    ucc = UseCaseContainer()
    ucc.tree_node_list.append(mxn_ps)

    mxn_ps_view = MxNTreeView(root=ucc)
    mxn_ps_view.selected_node = mxn_ps
    mxn_ps_view.replot = True
    mxn_ps_view.configure_traits()


def close():
    """ This closes the oricreate application.
    """
    oricreate.window.close()

if __name__ == '__main__':
    main()
