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

from traits.api import \
    Instance, HasStrictTraits,  \
    Dict,  WeakRef
from traitsui.api import \
    Item, View

from ibvpy.mats.mats_eval import IMATSEval


class MATSXDExplore(HasStrictTraits):
    '''
    Base class for MATSExplorer dimensional to manage dimensionally 
    dependent presentation of the material models.
    Simulate the loading histories of a material point in 1D space.
        '''
    explorer = WeakRef

    mats_eval = Instance(IMATSEval)

    explorer_config = Dict({})

    def _mats_eval_changed(self):
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

    traits_view = View(Item('mats_eval', show_label=False),
                       resizable=True,
                       width=1.0,
                       height=1.0
                       )
