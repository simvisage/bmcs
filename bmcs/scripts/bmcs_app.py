#!/usr/bin/env python
'''
Created on Dec 17, 2016

Toplevel script to start oricreate 

@author: rch
'''

import sys

from bmcs.bond_slip import \
    run_bond_slip_model_p, \
    run_bond_slip_model_d, \
    run_bond_slip_model_dp
from bmcs.pullout import \
    run_pullout_const_shear, \
    run_pullout_dp, run_pullout_multilinear
from traits.api import HasTraits, Button
from traitsui.api import View, UItem, VGroup, Group
from ibvpy.mats.mats3D.mats3D_plastic.yield_face3D_explorer \
    import run_explorer

# A bmcs mayavi instance so we can close it correctly.
bmcs = None


class BMCSLauncher(HasTraits):

    #=========================================================================
    # Lecture #2
    #=========================================================================
    bond_slip_model_d = Button(label='Damage')

    def _bond_slip_model_d_fired(self):
        run_bond_slip_model_d(kind='live')

    bond_slip_model_p = Button(label='Plasticity')

    def _bond_slip_model_p_fired(self):
        run_bond_slip_model_p(kind='live')

    bond_slip_model_dp = Button(label='Damage-plasticity')

    def _bond_slip_model_dp_fired(self):
        run_bond_slip_model_dp(kind='live')

    #=========================================================================
    # Lecture #3
    #=========================================================================
    pullout_model_const_shear = Button(label='Constant shear')

    def _pullout_model_const_shear_fired(self):
        run_pullout_const_shear(kind='live')

    pullout_model_multilinear = Button(label='Multi-linear')

    def _pullout_model_multilinear_fired(self):
        run_pullout_multilinear(kind='live')

    pullout_model_dp = Button(label='Damage-plasticity')

    def _pullout_model_dp_fired(self):
        run_pullout_dp(kind='live')

    #=========================================================================
    # Lecture #8
    #=========================================================================
    yield_face_explorer = Button(label='Yield conditions for concrete')

    def _yield_face_explorer_fired(self):
        run_explorer(kind='live')

    view = View(
        VGroup(
            Group(
                UItem('bond_slip_model_d',
                      full_size=True, resizable=True),
                UItem('bond_slip_model_p',
                      full_size=True, resizable=True),
                UItem('bond_slip_model_dp',
                      full_size=True, resizable=True),
                label='Bond-slip models, lecture #2'
            ),
            Group(
                UItem('pullout_model_const_shear',
                      full_size=True, resizable=True),
                UItem('pullout_model_multilinear',
                      full_size=True, resizable=True),
                UItem('pullout_model_dp',
                      full_size=True, resizable=True),
                label='Pull-out models, lecture #3'
            ),
            Group(
                UItem('yield_face_explorer',
                      full_size=True, resizable=True),
                label='Yield conditions, lecture #8'
            )
        ),
        title='BMCS application launcher',
        width=500,
        buttons=['OK']
    )


def run_bmcs_launcher():
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
    bmcs = BMCSLauncher()
    bmcs.configure_traits()


def close():
    """ This closes the oricreate application.
    """
    pass


if __name__ == '__main__':
    run_bmcs_launcher()
