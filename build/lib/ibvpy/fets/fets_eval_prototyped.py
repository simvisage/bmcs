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
# Created on Sep 15, 2009 by: rch

from traits.api import \
    Instance, \
    DelegatesTo

from ibvpy.core.tstepper_eval import \
    TStepperEval

from .i_fets_eval import IFETSEval


class FETSEvalPrototyped(TStepperEval):
    '''Base class of prototyped elements.

    These elements use the prototype fets for delivering the functionality.
    They may overload any of the prototype's traits and methods.
    '''
    prototype_fets = Instance(IFETSEval)

    # define all traits and methods as delegated to the prototype.

    dots_class = DelegatesTo('prototype_fets')
    dof_r = DelegatesTo('prototype_fets')
    geo_r = DelegatesTo('prototype_fets')
    n_nodal_dofs = DelegatesTo('prototype_fets')
    id_number = DelegatesTo('prototype_fets')
    mats_eval = DelegatesTo('prototype_fets')
    n_dof_r = DelegatesTo('prototype_fets')
    n_geo_r = DelegatesTo('prototype_fets')
    vtk_r = DelegatesTo('prototype_fets')
    vtk_cell_types = DelegatesTo('prototype_fets')
    vtk_cells = DelegatesTo('prototype_fets')
    vtk_node_cell_data = DelegatesTo('prototype_fets')
    vtk_ip_cell_data = DelegatesTo('prototype_fets')
    n_vtk_r = DelegatesTo('prototype_fets')
    n_vtk_cells = DelegatesTo('prototype_fets')
    vtk_pnt_ip_map = DelegatesTo('prototype_fets')

    ip_coords = DelegatesTo('prototype_fets')
    ip_weights = DelegatesTo('prototype_fets')
    get_ip_scheme = DelegatesTo('prototype_fets')
    n_gp = DelegatesTo('prototype_fets')
    vtk_r_arr = DelegatesTo('prototype_fets')

    def get_ip_scheme(self, *params):
        return self.prototype_fets.get_ip_scheme(*params)

    def get_vtk_pnt_ip_map_data(self, vtk_r):
        return self.prototype_fets.get_vtk_pnt_ip_map_data(vtk_r)

    def get_vtk_r_glb_arr(self, X_mtx, r_mtx=None):
        return self.prototype_fets.get_vtk_r_glb_arr(X_mtx, r_mtx)

    def get_X_pnt(self, sctx):
        return self.prototype_fets.get_X_pnt(sctx)

    def map_r2X(self, r_pnt, X_mtx):
        return self.map_r2X(r_pnt, X_mtx)

    n_e_dofs = DelegatesTo('prototype_fets')

    dim_slice = DelegatesTo('prototype_fets')

    def new_cntl_var(self):
        self.new_cntl_var()

    def new_resp_var(self):
        self.new_resp_var()

    m_arr_size = DelegatesTo('prototype_fets')
    _get_m_arr_size = DelegatesTo('prototype_fets')

    def get_mp_state_array_size(self, sctx):
        self.get_mp_state_array_size(sctx)

    ngp_r = DelegatesTo('prototype_fets')
    ngp_s = DelegatesTo('prototype_fets')
    ngp_t = DelegatesTo('prototype_fets')

    debug_on = DelegatesTo('prototype_fets')
    _debug_rte_dict = DelegatesTo('prototype_fets')
    rte_dict = DelegatesTo('prototype_fets')

    traits_view = DelegatesTo('prototype_fets')

    def adjust_spatial_context_for_point(self, sctx):
        print('YYYYYYYYYYYYYYYYYY')
        self.prototype_fets.adjust_spatial_context_for_point(sctx)

    def get_state_array_size(self):
        return self.prototype_fets.get_state_array_size()

    def setup(self, sctx):
        self.prototype_fets.setup(sctx)

    def get_corr_pred(self, sctx, u, du, tn, tn1, u_avg=None,
                      B_mtx_grid=None, J_det_grid=None,
                      ip_coords=None, ip_weights=None):
        return self.prototype_fets.get_corr_pred(sctx, u, du, tn, tn1, u_avg, B_mtx_grid, J_det_grid, ip_coords, ip_weights)

    def get_J_mtx(self, r_pnt, X_mtx):
        return self.prototype_fets.get_J_mtx(r_pnt, X_mtx)

    def get_N_geo_mtx(self, r_pnt):
        return self.prototype_fets.get_N_geo_mtx(r_pnt)

    def get_N_mtx(self, r_pnt):
        return self.prototype_fets.get_N_mtx(r_pnt)

    def get_dNr_geo_mtx(self, r_pnt):
        return self.prototype_fets.get_dNr_geo_mtx(r_pnt)

    def get_B_mtx(self, r_pnt, X_mtx):
        return self.prototype_fets.get_B_mtx(r_pnt, X_mtx)

    def get_mtrl_corr_pred(self, sctx, eps_mtx, d_eps, tn, tn1, eps_avg=None):
        return self.prototype_fets.get_mtrl_corr_pred(sctx, eps_mtx, d_eps, tn, tn1, eps_avg)

    def get_J_det(self, r_pnt, X_mtx):
        return self.prototype_fets.get_J_det(r_pnt, X_mtx)

    def _get_J_det(self, r_pnt3d, X_mtx):
        return self.prototype_fets._get_J_det(r_pnt3d, X_mtx)

    def get_eps_eng(self, sctx, u):
        return self.prototype_fets.get_eps_eng(sctx, u)

    def get_u(self, sctx, u):
        return self.get_u(sctx, u)
