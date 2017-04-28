#!/usr/bin/env python
'''
Created on Dec 17, 2016

Toplevel script to start oricreate 

@author: rch
'''

import sys


# A bmcs mayavi instance so we can close it correctly.
bmcs = None

from bmcs.bond_slip import \
    run_bond_slip_model_p, \
    run_bond_slip_model_d, \
    run_bond_slip_model_dp


def run_bond_app(run_bond_slip_model):
    """This starts up the oricreate application.
    """
    global bmcs

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

    run_bond_slip_model()


def close():
    """ This closes the oricreate application.
    """
    pass


def run_d():
    run_bond_app(run_bond_slip_model_d)


def run_p():
    run_bond_app(run_bond_slip_model_p)


def run_dp():
    run_bond_app(run_bond_slip_model_dp)

if __name__ == '__main__':
    run_dp()
