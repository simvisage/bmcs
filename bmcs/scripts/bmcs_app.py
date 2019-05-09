#!/usr/bin/env python
'''
Created on Dec 17, 2016

Toplevel script to start BMCS 

@author: rch
'''

import sys

from bmcs.bond_slip import \
    run_bond_sim_damage, \
    run_bond_sim_elasto_plasticity
from bmcs.crack_mode_I import \
    run_bending3pt_mic_odf, \
    run_bending3pt_sdamage_viz2d, \
    run_bending3pt_sdamage_viz3d, \
    run_tensile_test_sdamage
from bmcs.pullout import \
    run_pullout_const_shear, \
    run_pullout_dp, \
    run_pullout_ep_cyclic, \
    run_pullout_multilinear, \
    run_pullout_frp_damage, \
    run_pullout_fatigue
from ibvpy.mats.mats3D.mats3D_plastic.yield_face3D_explorer import run_explorer
from traits.api import HasTraits, Button, Str, Constant
from traitsui.api import \
    View, Item, UItem, VGroup, Group, Spring, HGroup

from ibvpy.mats.mats3D.mats3D_plastic.yield_face3D_explorer \
    import run_explorer

from .bmcs_version import CURRENT_VERSION


# A bmcs mayavi instance so we can close it correctly.
bmcs = None


class BMCSLauncher(HasTraits):

    version = Constant(CURRENT_VERSION)
    #=========================================================================
    # Lecture #2
    #=========================================================================
    bond_slip_model_d = Button(label='Damage')

    def _bond_slip_model_d_fired(self):
        run_bond_sim_damage()

    bond_slip_model_p = Button(label='Plasticity')

    def _bond_slip_model_p_fired(self):
        run_bond_sim_elasto_plasticity()

    bond_slip_model_dp = Button(label='Damage-plasticity')

    def _bond_slip_model_dp_fired(self):
        pass
        # run_bond_slip_model_dp(kind='live')

    #=========================================================================
    # Lecture #3
    #=========================================================================
    pullout_model_const_shear = Button(label='Constant shear')

    def _pullout_model_const_shear_fired(self):
        run_pullout_const_shear(kind='live')

    pullout_model_multilinear = Button(label='Multi-linear')

    def _pullout_model_multilinear_fired(self):
        run_pullout_multilinear(kind='live')

    pullout_model_frp_damage = Button(label='FRP damage')

    def _pullout_model_frp_damage_fired(self):
        run_pullout_frp_damage(kind='live')

    pullout_model_ep = Button(label='Elasto-plasticity')

    def _pullout_model_ep_fired(self):
        run_pullout_ep_cyclic()

    pullout_model_dp = Button(label='Damage-plasticity')

    def _pullout_model_dp_fired(self):
        run_pullout_dp(kind='live')

    pullout_model_fatigue = Button(label='Damage-fatigue')

    def _pullout_model_fatigue_fired(self):
        run_pullout_fatigue(kind='live')

    #=========================================================================
    # Lecture #8
    #=========================================================================

    tensile_test_2d_sdamage = Button(label='Tensile test - isotropic damage')

    def _tensile_test_2d_sdamage_fired(self):
        run_tensile_test_sdamage(kind='live')

    bending3pt_3d = Button(label='Bending test (3D)')

    def _bending3pt_3d_fired(self):
        run_bending3pt_mic_odf(kind='live')

    bending3pt_2d_sdamage_viz2d = Button(
        label='bending test 3Pt - isotropic damage (2D-light)')

    def _bending3pt_2d_sdamage_viz2d_fired(self):
        run_bending3pt_sdamage_viz2d(kind='live')

    bending3pt_2d_sdamage_viz3d = Button(
        label='Bending test 3Pt - isotropic damage (2D-heavy)')

    def _bending3pt_2d_sdamage_viz3d_fired(self):
        run_bending3pt_sdamage_viz3d(kind='live')

    #=========================================================================
    # Lecture #6
    #=========================================================================
    yc_explorer = Button(
        label='Yield conditions for concrete')

    def _yc_explorer_fired(self):
        run_explorer(kind='live')

    view = View(
        VGroup(
            HGroup(
                Spring(),
                Item('version', style='readonly',
                     full_size=True, resizable=True,
                     )
            ),
            Group(
                UItem('bond_slip_model_d',
                      full_size=True, resizable=True,
                      enabled_when='True'
                      ),
                UItem('bond_slip_model_p',
                      full_size=True, resizable=True,
                      enabled_when='True'
                      ),
                UItem('bond_slip_model_dp',
                      full_size=True, resizable=True,
                      enabled_when='False'
                      ),
                label='Bond-slip models, lecture #1-2'
            ),
            Group(
                UItem('pullout_model_const_shear',
                      full_size=True, resizable=True,
                      enabled_when='True'
                      ),
                UItem('pullout_model_multilinear',
                      full_size=True, resizable=True,
                      enabled_when='True'
                      ),
                UItem('pullout_model_ep',
                      full_size=True, resizable=True,
                      enabled_when='True'
                      ),
                UItem('pullout_model_frp_damage',
                      full_size=True, resizable=True,
                      enabled_when='True'
                      ),
                UItem('pullout_model_dp',
                      full_size=True, resizable=True,
                      enabled_when='True'
                      ),
                UItem('pullout_model_fatigue',
                      full_size=True, resizable=True,
                      enabled_when='True'
                      ),
                label='Pull-out models, lecture #3-6'
            ),
            Group(
                UItem('tensile_test_2d_sdamage',
                      full_size=True, resizable=True,
                      enabled_when='False'
                      ),
                UItem('bending3pt_2d_sdamage_viz2d',
                      full_size=True, resizable=True,
                      enabled_when='False'
                      ),
                UItem('bending3pt_2d_sdamage_viz3d',
                      full_size=True, resizable=True,
                      enabled_when='False'
                      ),
                UItem('bending3pt_3d',
                      full_size=True, resizable=True,
                      enabled_when='False'
                      ),
                label='Bending, crack propagation, lecture #7-9'
            ),
            Group(
                UItem('yc_explorer',
                      full_size=True, resizable=True),
                label='Yield surface explorer #10'
            ),
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
