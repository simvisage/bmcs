#-------------------------------------------------------------------------
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
# Created on Sep 4, 2009 by: rch

from ibvpy.core.bcond_mngr import BCondMngr
from ibvpy.core.tstepper import TStepper
from ibvpy.mats.mats_eval import IMATSEval
#from pyglet.media.drivers.alsa.asound import u_int16_t
from traits.api import \
    Instance, \
    Dict,  WeakRef, List, implements, \
    DelegatesTo, Bool
from traitsui.api import \
    Item, View
from view.ui import BMCSTreeNode

import numpy as np


class MATSXDExplore(TStepper):

    '''
    Base class for MATSExplorer dimensional to manage dimensionally 
    dependent presentation of the material models.
    Simulate the loading histories of a material point in 1D space.
        '''
    node_name = 'Stress space'

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [
            self.mats_eval
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.mats_eval
        ]

    algorithmic_stiffness = Bool(
        True, ALG=True, auto_set=False, enter_set=True)

    state_array_shapes = DelegatesTo('mats_eval')

    explorer = WeakRef

    mats_eval = Instance(IMATSEval)

    # Boundary condition manager
    #
    bcond_mngr = Instance(BCondMngr)

    def _bcond_mngr_default(self):
        return BCondMngr()

    explorer_config = Dict({})

    def _mats_eval_changed(self):
        return
        if self.explorer_config:
            ec = self.explorer_config
        else:
            ec = self.mats_eval.explorer_config
        mats_eval = ec.get('mats_eval', self.mats_eval)
        if self.explorer == None:
            return

        self.explorer.tloop.tstepper.tse = mats_eval
        self.explorer.tloop.tstepper.sdomain.mats_eval = mats_eval

        tl = self.explorer.tloop
        tl.bcond_list = ec['bcond_list']
        tl.rtrace_list = ec['rtrace_list']
        if ec.has_key('tline'):
            tl.tline = ec['tline']
        tl.reset()

    def new_cntl_var(self):
        '''
        Return contoll variable array
        '''
        return np.zeros(6, np.float_)

    def new_resp_var(self):
        '''
        Return control response array
        '''
        return np.zeros(6, np.float_)

    def get_corr_pred(self, U, dU, t_n, t_n1, update_state,
                      **state_vars):
        eps_Emab = self.mats_eval.map_eps_eng_to_mtx(U)[np.newaxis, ...]
        deps_Emab = self.mats_eval.map_eps_eng_to_mtx(dU)[np.newaxis, ...]
        D_Emabef, sig_Emab = self.mats_eval.get_corr_pred(
            eps_Emab, deps_Emab, t_n, t_n1,
            update_state, self.algorithmic_stiffness,
            **state_vars
        )
        K = self.mats_eval.map_tns4_to_tns2(D_Emabef[0])
        F_int = self.mats_eval.map_sig_mtx_to_eng(sig_Emab[0])
        return K, F_int

    traits_view = View(Item('mats_eval', show_label=False),
                       resizable=True,
                       width=1.0,
                       height=1.0
                       )

    tree_view = traits_view
