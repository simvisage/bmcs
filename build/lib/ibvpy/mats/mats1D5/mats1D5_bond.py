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
# Created on Sep 8, 2009 by: rch
from envisage.ui.workbench.api import WorkbenchApplication
from ibvpy.mats.mats1D import \
    MATS1DElastic, MATS1DPlastic, MATS1DDamage
from ibvpy.mats.mats1D5.mats1D5_eval import MATS1D5Eval
from ibvpy.mats.mats_eval import IMATSEval
from numpy import zeros, zeros_like, array
from traits.api import provides, \
    Instance, Property, cached_property, List, \
    Callable, String, Int, HasTraits
from traitsui.api import \
    View, Item
from util.traits.either_type import \
    EitherType

mats_klasses = [MATS1DElastic,
                MATS1DPlastic,
                MATS1DDamage]


@provides(IMATSEval)
class MATS1D5Bond(MATS1D5Eval):

    '''Bond model for two phases interacting over an interface with zero thickness.

    Both phases can be associated with arbitrary 1D mats model.
    Interface behavior can be defined for both sliding and opening using 1D mats models.
    '''

    #-------------------------------------------------------------------------
    # Submodels constituting the interface behavior
    #-------------------------------------------------------------------------

    mats_phase1 = EitherType(klasses=mats_klasses,
                             desc='Material model of phase 1')
    mats_ifslip = EitherType(klasses=mats_klasses,
                             desc='Material model for interface slippage')
    mats_ifopen = EitherType(klasses=mats_klasses,
                             desc='Material model for interface opening')
    mats_phase2 = EitherType(klasses=mats_klasses,
                             desc='Material model for phase 2')

    def _mats_phase1_default(self):
        return MATS1DElastic()

    def _mats_ifslip_default(self):
        return MATS1DElastic()

    def _mats_ifopen_default(self):
        return MATS1DElastic()

    def _mats_phase2_default(self):
        return MATS1DElastic()

    traits_view = View(Item('mats_phase1@'),
                       Item('mats_ifslip@'),
                       Item('mats_ifopen@'),
                       Item('mats_phase2@'),
                       resizable=True,
                       scrollable=True,
                       width=0.8,
                       height=0.9,
                       buttons=['OK', 'Cancel'])

    #-------------------------------------------------------------------------
    # Subsidiary maps to enable generic loop over the material models
    #-------------------------------------------------------------------------

    # order of the material model names used for association in generic loops
    _mats_names = List(
        ['mats_phase1', 'mats_slip', 'mats_open', 'mats_phase2'])

    # hidden list of all four involved mats
    _mats_list = Property(
        depends_on='mats_phase1, mats_slip, mats_open, mats_phase2')

    @cached_property
    def _get__mats_list(self):
        return [self.mats_phase1, self.mats_ifslip, self.mats_ifopen, self.mats_phase2]

    # state array map with the sizes of the array
    _state_sizes = Property(
        depends_on='mats_phase1, mats_slip, mats_open, mats_phase2')

    @cached_property
    def _get__state_sizes(self):
        return array([mats.get_state_array_size()
                      for mats in self._mats_list], dtype='int_')

    # state array map with the sizes of the array
    _state_offsets = Property(
        depends_on='mats_phase1, mats_slip, mats_open, mats_phase2')

    @cached_property
    def _get__state_offsets(self):
        offsets = zeros_like(self._state_sizes)
        offsets[1:] = self._state_sizes[:-1].cumsum()
        return offsets

    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------

    def get_state_array_size(self):
        '''
        Return the number of floats to be saved
        '''
        return self._state_sizes.cumsum()[-1]

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps_app_eng, tn, tn1, *args, **kw):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''

        sig_app_eng = zeros_like(eps_app_eng)
        # @todo [rch] dirty - when called form response tracer
        if isinstance(d_eps_app_eng, int) and d_eps_app_eng == 0:
            d_eps_app_eng = zeros_like(eps_app_eng)

        D_mtx = zeros(
            (eps_app_eng.shape[0], eps_app_eng.shape[0]), dtype='float_')

        mats_state_array = sctx.mats_state_array

        for i, mats in enumerate(self._mats_list):

            # extract the stress components
            eps, d_eps = eps_app_eng[i], d_eps_app_eng[i]
            size = self._state_sizes[i]
            offset = self._state_offsets[i]
            sctx.mats_state_array = mats_state_array[offset: offset + size]

            sig_app_eng[i], D_mtx[i, i] = mats.get_corr_pred(
                sctx, eps, d_eps, tn, tn1)

        sctx.mats_state_array = mats_state_array

        return sig_app_eng, D_mtx

    def get_sig1(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return sig_eng[0:1]

    def get_sig2(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return sig_eng[3:]

    def get_shear_flow(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return sig_eng[1:2]

    def get_cohesive_stress(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return sig_eng[2:3]

    rte_dict = Property

    def _get_rte_dict(self):

        rte_dict = {}
        ix_maps = [0, 1, 2, 3]
        for name, mats, ix_map, size, offset in \
            zip(self._mats_names, self._mats_list, ix_maps, self._state_sizes,
                self._state_offsets):
            for key, v_eval in list(mats.rte_dict.items()):

                __call_v_eval = RTE1D5Bond(v_eval=v_eval,
                                           name=name + '_' + key,
                                           size=size,
                                           offset=offset,
                                           ix_map=ix_map)

                rte_dict[name + '_' + key] = __call_v_eval

        # sigma is also achievable through phase1_sig_app and phase_2_sig_app
        extra_rte_dict = {'sig1': self.get_sig1,
                          'sig2': self.get_sig2,
                          'shear_flow': self.get_shear_flow,
                          'cohesive_stress': self.get_cohesive_stress,
                          }
        rte_dict.update(extra_rte_dict)
        return rte_dict
    #-------------------------------------------------------------------------
    # Methods required by the mats_explore tool
    #-------------------------------------------------------------------------

    def new_cntl_var(self):
        return zeros(4, 'float_')

    def new_resp_var(self):
        return zeros(4, 'float_')


class RTE1D5Bond(HasTraits):

    v_eval = Callable
    name = String
    size = Int
    offset = Int
    ix_map = Int

    def __call__(self, sctx, u, *args, **kw):
        u_x = array([u[self.ix_map]], dtype='float')
        # save the spatial context
        mats_state_array = sctx.mats_state_array
        sctx.mats_state_array = mats_state_array[
            self.offset: self.offset + self.size]
        result = self.v_eval(sctx, u_x, *args, **kw)
        # put the spatial context back
        sctx.mats_state_array = mats_state_array

        return result


if __name__ == '__main__':

    mats = MATS1D5Bond()
    mats.configure_traits()
