'''
Created on May 20, 2009

@author: jakub
'''
from ibvpy.mats.mats2D.mats2D_tensor import map2d_eps_eng_to_mtx
from ibvpy.mats.mats3D.mats3D_tensor import map3d_eps_eng_to_mtx
from mathkit.matrix_la.sys_mtx_array import SysMtxArray
from numpy import ix_, frompyfunc, array, abs, vstack, linalg, dot, ones, hstack, \
    arange, zeros_like, zeros, isinf, where, copy
from traits.api import \
    Array, Bool, Float, \
    Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
    Tuple, Property, cached_property, Dict

from tvtk.api import tvtk

from tvtk.tvtk_classes import tvtk_helper

from .dots_eval import DOTSEval, RTraceEvalUDomainFieldVar


class XDOTSEval(DOTSEval):

    '''
    Domain with uniform FE-time-step-eval.
    '''
    elem_r_contours = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_elem_r_contours(self):
        '''
        Get coordinates of vertices grouped in three subsets
        1) vertices with positive value of level set function
        2) vertices with negative value of level set function
        3) intersection of the level set with the element edges   

        This serves as an input for the triangulation

        It has to be done in the loop as the number of points changes for the elems
        Works for 1 and 2D
        '''

        corner_dof_r = array(self.fets_eval.dof_r)[self.dof_r_corner_idx]

        i_r = self.sdomain.ls_intersection_r

        # print 'dof r corner ',self.dof_r_corner_idx
        pos_r = []
        neg_r = []
        dn_ls_val = self.dof_node_ls_values[:, self.dof_r_corner_idx]
        for dof_vals in dn_ls_val:
            pos_r.append(corner_dof_r[dof_vals.flatten() > 0.])
            neg_r.append(corner_dof_r[dof_vals.flatten() < 0.])
        return [pos_r, neg_r, i_r]

    # @todo - use the4 vertex_X_map available in sdomain
    dof_r_corner_idx = Property(Array(bool))

    @cached_property
    def _get_dof_r_corner_idx(self):
        '''
        Extracts indices of the corner dofs for visualization
        Works for 1 and 2D

        @todo - this is done in cell_grid_spec - use the method from there
        '''
        dof_r = array(self.fets_eval.dof_r)
        abs_dof_r = abs(dof_r)
        if abs_dof_r.shape[1] == 1:
            return (abs_dof_r[:] == 1.).flatten()
        elif abs_dof_r.shape[1] == 2:
            return (abs_dof_r[:, 0] * abs_dof_r[:, 1]) == 1.

    elem_triangulation = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_elem_triangulation(self):
        '''Get discretization for integration over the element.

        In 1D - the output is a set of line segments
        In 2D - a list of triangles is returned
        In 3D - does not work - probably some topological info needs to be added.
        '''
        division = []
        for pos_r, neg_r, i_r in zip(self.elem_r_contours[0],
                                     self.elem_r_contours[1],
                                     self.elem_r_contours[2]):  # TODO:this can be done better
            point_set = [vstack((pos_r, i_r)),
                         vstack((neg_r, i_r))]
            division.append(self.fets_eval.get_triangulation(point_set))
        return division

    def get_eps(self, sctx, u):
        e_id = sctx.e_id
        p_id = sctx.p_id
        X_mtx = sctx.X
        r_pnt = sctx.loc
        B_mtx = self.fets_eval.get_B_mtx(r_pnt, X_mtx,
                                         self.dof_node_ls_values[e_id],
                                         self.vtk_ls_values[e_id][p_id])
        eps = dot(B_mtx, u)
        shape = eps.shape[0]
        if shape == 1:
            return eps
        elif shape == 3:
            return map2d_eps_eng_to_mtx(eps)
        elif shape == 6:
            return map3d_eps_eng_to_mtx(eps)

    def get_eps_m(self, sctx, u):
        e_id = sctx.e_id
        p_id = sctx.p_id
        X_mtx = sctx.X
        r_pnt = sctx.loc
        B_mtx = self.fets_eval.get_B_mtx(r_pnt, X_mtx,
                                         self.dof_node_ls_values[e_id],
                                         self.vtk_ls_values[e_id][p_id])
        eps = dot(B_mtx, u)
        return array([[eps[0], eps[2]], [eps[2], eps[1]]])

    def get_eps_f(self, sctx, u):
        e_id = sctx.e_id
        p_id = sctx.p_id
        X_mtx = sctx.X
        r_pnt = sctx.loc
        B_mtx = self.fets_eval.get_B_mtx(r_pnt, X_mtx,
                                         self.dof_node_ls_values[e_id],
                                         self.vtk_ls_values[e_id][p_id])
        eps = dot(B_mtx, u)
        return array([[eps[3], eps[5]], [eps[5], eps[4]]])

    def get_u(self, sctx, u):
        e_id = sctx.e_id
        p_id = sctx.p_id
        N_mtx = self.fets_eval.get_N_mtx(sctx.loc,
                                         self.dof_node_ls_values[e_id],
                                         self.vtk_ls_values[e_id][p_id])
#        print "N ",N_mtx
#        print "u ",u
#        print "x u",dot( N_mtx, u )
        return dot(N_mtx, u)

    def get_u_m(self, sctx, u):
        e_id = sctx.e_id
        p_id = sctx.p_id
        N_mtx = self.fets_eval.get_N_mtx(sctx.loc,
                                         self.dof_node_ls_values[e_id],
                                         self.vtk_ls_values[e_id][p_id])
        return dot(N_mtx, u)[:2]

    def get_u_rf(self, sctx, u):
        e_id = sctx.e_id
        p_id = sctx.p_id
        N_mtx = self.fets_eval.get_N_mtx(sctx.loc,
                                         self.dof_node_ls_values[e_id],
                                         self.vtk_ls_values[e_id][p_id])
        return dot(N_mtx, u)[1:2]

    def get_u_rm(self, sctx, u):
        e_id = sctx.e_id
        p_id = sctx.p_id
        N_mtx = self.fets_eval.get_N_mtx(sctx.loc,
                                         self.dof_node_ls_values[e_id],
                                         self.vtk_ls_values[e_id][p_id])
        return dot(N_mtx, u)[:1]

    def get_u_f(self, sctx, u):
        e_id = sctx.e_id
        p_id = sctx.p_id
        N_mtx = self.fets_eval.get_N_mtx(sctx.loc,
                                         self.dof_node_ls_values[e_id],
                                         self.vtk_ls_values[e_id][p_id])
        return dot(N_mtx, u)[2:]

    def map_u(self, sctx, U):
        ix = sctx.elem.get_dof_map()
        sctx.dots = self  # todo: this is ugly
#        sctx.r = fets_eval.map_to_local( sctx.elem, sctx.X )
        u = U[ix]
        return u

    rte_dict = Property(Dict, depends_on='fets_eval')

    @cached_property
    def _get_rte_dict(self):
        rte_dict = {}

        rte_dict.update({'eps': RTraceEvalUDomainFieldVar(eval=self.get_eps, ts=self, u_mapping=self.map_u),
                         'u': RTraceEvalUDomainFieldVar(eval=self.get_u, ts=self, u_mapping=self.map_u),
                         'u_m': RTraceEvalUDomainFieldVar(eval=self.get_u_m, ts=self, u_mapping=self.map_u),
                         'u_f': RTraceEvalUDomainFieldVar(eval=self.get_u_f, ts=self, u_mapping=self.map_u),
                         'eps_m': RTraceEvalUDomainFieldVar(eval=self.get_eps_m, ts=self, u_mapping=self.map_u),
                         'eps_f': RTraceEvalUDomainFieldVar(eval=self.get_eps_f, ts=self, u_mapping=self.map_u),
                         'u_rm': RTraceEvalUDomainFieldVar(eval=self.get_u_rm, ts=self, u_mapping=self.map_u),
                         'u_rf': RTraceEvalUDomainFieldVar(eval=self.get_u_rf, ts=self, u_mapping=self.map_u), })
        for key, eval in list(self.fets_eval.rte_dict.items()):
            rte_dict[key] = RTraceEvalUDomainFieldVar(name=key,
                                                      u_mapping=self.map_u,
                                                      eval=eval,
                                                      fets_eval=self.fets_eval)
        return rte_dict

    rt_triangles = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_rt_triangles(self):
        triangles = []
        for pos_r, neg_r, i_r in zip(self.elem_r_contours[0],
                                     self.elem_r_contours[1],
                                     self.elem_r_contours[2]):  # TODO:this can be done better
            if i_r.shape[1] == 1:
                shift = self.sdomain.rt_tol
                norm_vct = 1

            elif i_r.shape[1] == 2:
                # direction vector of the intersection
                dir_vct = i_r[1] - i_r[0]
                # normal_vector
                norm_vct = array([-dir_vct[1], dir_vct[0]], dtype=float)
                shift_vct = norm_vct / \
                    linalg.norm(norm_vct) * self.sdomain.rt_tol
                # ???
                delta_rs = array([shift_vct[1] ** 2 / shift_vct[0],
                                  shift_vct[0] ** 2 / shift_vct[1]], dtype=float)
                # check for zero division
                i_inf = isinf(delta_rs)
                delta_rs[i_inf] = zeros(2)[i_inf]
                shift_rs = shift_vct + delta_rs
                # TODO:make for arbitrary number of i_r
                shift_rs_pts = vstack((shift_rs, shift_rs))
                # print 'new directions ',shift_rs
            pos_pts = i_r.copy()
            neg_pts = i_r.copy()
            # check which coordinate lies not on the element edge
            edge_idx = where(abs(i_r) != 1)

            pos_dir = pos_r[0] - i_r[0]  # TODO:generalize also for 1d
            # check that the normal points out from the crack
            if dot(pos_dir, norm_vct) > 0.:
                #pos_pts =  i_r + shift_vct
                #neg_pts =  i_r - shift_vct
                pos_pts[edge_idx] += shift_rs_pts[edge_idx]
                neg_pts[edge_idx] -= shift_rs_pts[edge_idx]
            else:
                #pos_pts =  i_r - shift_vct
                #neg_pts =  i_r + shift_vct
                pos_pts[edge_idx] -= shift_rs_pts[edge_idx]
                neg_pts[edge_idx] += shift_rs_pts[edge_idx]

            if self.sdomain.ls_side_tag == 'both':
                point_set = [vstack((pos_r, pos_pts)),
                             vstack((neg_r, neg_pts))]
            elif self.sdomain.ls_side_tag == 'pos':
                point_set = [vstack((pos_r, pos_pts))]
            elif self.sdomain.ls_side_tag == 'neg':
                point_set = [vstack((neg_r, neg_pts))]
            triangles.append(self.fets_eval.get_triangulation(point_set))
        return triangles

    ip_X = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_ip_X(self):
        '''
        Integration points in global coords 3D needed for evaluation of ls values,
        can be also used for postprocesing
        '''
        ip_X = []
        for X_mtx, ip_addr0, ip_addr1 in zip(self.sdomain.elem_X_map,
                                             self.ip_offset[:-1],
                                             self.ip_offset[1:]):
            ip_slice = slice(ip_addr0, ip_addr1)
            ip_X.append(
                self.fets_eval.get_vtk_r_glb_arr(X_mtx, self.ip_coords[ip_slice]))
        return vstack(ip_X)

    ip_offset = Property

    def _get_ip_offset(self):
        return self.integ_structure[0]

    ip_coords = Property

    def _get_ip_coords(self):
        return self.integ_structure[1]

    ip_weights = Property

    def _get_ip_weights(self):
        return self.integ_structure[2]

    state_start_elem_grid = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_state_start_elem_grid(self):
        # create the element grid to store the offsets of the elements
        state_elem_grid = self.sdomain.intg_elem_grid.copy()
        elem_grid_ix = self.sdomain.intg_grid_ix
        state_elem_grid[elem_grid_ix] = self.ip_offset[:-1]
        return state_elem_grid

    state_end_elem_grid = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_state_end_elem_grid(self):
        # create the element grid to store the offsets of the elements
        state_elem_grid = self.sdomain.intg_elem_grid.copy()
        elem_grid_ix = self.sdomain.intg_grid_ix
        state_elem_grid[elem_grid_ix] = self.ip_offset[1:]
        return state_elem_grid

    integ_structure = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_integ_structure(self):
        ip_off_list = [0]
        ip_coo_list = []
        ip_wei_list = []
        ip_offset = 0

        for elem_triangles in self.elem_triangulation:
            ip_coords = self.fets_eval.get_ip_coords(elem_triangles,
                                                     self.fets_eval.int_order)
            ip_offset += ip_coords.shape[0]
            ip_off_list.append(ip_offset)
            ip_coo_list.append(ip_coords)
            ip_wei_list.append(self.fets_eval.get_ip_weights(elem_triangles,
                                                             self.fets_eval.int_order))
        ip_off_arr = array(ip_off_list, dtype=int)

        # handle the case of empty domain
        if len(ip_coo_list) == 0:
            raise ValueError(
                'empty subdomain - something wrong in the fe_domain management')

        ip_coo_arr = vstack(ip_coo_list)[:, self.fets_eval.dim_slice]
        ip_w_arr = hstack(ip_wei_list)

        return (ip_off_arr,
                ip_coo_arr,
                ip_w_arr)

    # Integration over discontinuity
    # discretize the integration points lying in the level set
    disc_integ_structure = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_disc_integ_structure(self):
        # nip_disc - number of integration points along the discontinuity
        if self.fets_eval.nip_disc == 0:
            return array([zeros_like(self.ip_offset)], dtype=int)
        ip_off_list = [0]
        ip_coo_list = []
        ip_wei_list = []
        ip_offset = 0
        id_map = array(
            [arange(self.sdomain.fe_grid_slice.r_i.shape[1])], dtype=int)

        # @todo - the r_i in fe_grid_slice ignores the boundary in the FEXSubdomain
        # change - to remove the irrelevant elements beyound the boundary.

        for segment in self.sdomain.fe_grid_slice.r_i:
            ip_coords = self.fets_eval.get_ip_coords(
                [segment, id_map], self.fets_eval.nip_disc)
            ip_offset += ip_coords.shape[0]
            ip_off_list.append(ip_offset)
            ip_coo_list.append(ip_coords)
            ip_wei_list.append(
                self.fets_eval.get_ip_weights([segment, id_map], self.fets_eval.nip_disc))
        return array(ip_off_list, dtype=int), vstack(ip_coo_list)[:, self.fets_eval.dim_slice], vstack(ip_wei_list).flatten()


#        if self.fets_eval.nip_disc:
#            ip_off_list = [0,1]
#        else:
# ip_off_list = zeros_like(self.ip_offset)#necessary for consistent interface in corr pred
#
# ip_coord = self.sdomain.fe_grid_slice.r_i.flatten()#this can work just in 1D
# return array(ip_off_list),vstack(ip_coo_list)[:, self.fets_eval.dim_slice],vstack(ip_wei_list).flatten()
# return array(ip_off_list , dtype =
# int),array([ip_coord],dtype=float),array([2.],dtype=float)

    ip_disc_coords = Property

    def _get_ip_disc_coords(self):
        return self.disc_integ_structure[1]

    ip_disc_weights = Property

    def _get_ip_disc_weights(self):
        return self.disc_integ_structure[2]

    ip_disc_offset = Property

    def _get_ip_disc_offset(self):
        return self.disc_integ_structure[0]

    ip_ls_values = Property(Array(Float))

    def _get_ip_ls_values(self):
        # TODO:define the ineraction wirh ls
        ls_fn = frompyfunc(self.sdomain.ls_fn_X, 2, 1)

        X, Y, Z = self.ip_X.T  # 3d coords - vtk
        return ls_fn(X, Y).flatten()

    ip_normal = Property(Array(Float))

    def _get_ip_normal(self):
        ir_shape = self.sdomain.ls_intersection_r.shape[2]
        if ir_shape == 1:  # 1D
            # assuming that the first node is the left-most
            if self.dof_node_ls_values[0, 0] < 0.:
                normal = [1.]
            else:
                normal = [-1.]
        elif ir_shape == 2:  # 2D
            for i_r in self.sdomain.ls_intersection_r:
                # direction vector of the intersection
                dir_vct = i_r[1] - i_r[0]
                normal = array([-dir_vct[1], dir_vct[0]])  # normal_vector
        return normal

    dof_node_ls_values = Property(Array(Float))

    def _get_dof_node_ls_values(self):
        return self.sdomain.dof_node_ls_values

    def get_vtk_X(self, position):
        return vstack(self.vtk_X)

    vtk_X = Property(
        depends_on='sdomain.changed_structure, sdomain.+changed_geometry')

    @cached_property
    def _get_vtk_X(self):
        '''Get the discretization points based on the fets_eval 
        associated with the current domain.
        '''
        vtk_X = []
        for triangle, e in zip(self.rt_triangles, self.sdomain.elements):
            X_mtx = e.get_X_mtx()
            # TODO:slicing works just for 2D
            vtk_X.append(
                self.fets_eval.get_vtk_r_glb_arr(X_mtx, triangle[0][:, :2]))
        # print 'vtk_X ',vtk_X
        # return array(vtk_X)
        return vtk_X  # have to stay list for arbitraly number of pts

    debug_cell_data = Bool(False)

    def get_vtk_cell_data(self, position, point_offset, cell_offset):
        cells = []
        for triangle in self.rt_triangles:  # TODO:offset can be done simpler
            cells.append(triangle[1] + point_offset)
            point_offset += triangle[0].shape[0]
        vtk_cells = vstack(cells)
#        vtk_cells = vstack([triangle[1]
#                     for triangle in self.rt_triangles])
        # print "vtk_cells_array", vtk_cells
        n_cell_points = vtk_cells.shape[1]
        n_cells = vtk_cells.shape[0]
        vtk_cell_array = hstack((ones((n_cells, 1), dtype=int) * n_cell_points,
                                 vtk_cells))
        vtk_cell_offsets = arange(
            n_cells, dtype=int) * (n_cell_points + 1) + cell_offset
        if n_cell_points == 3:
            cell_str = 'Triangle'
        elif n_cell_points == 2:
            cell_str = 'Line'
        cell_class = tvtk_helper.get_class(cell_str)
        cell_type = cell_class().cell_type
        vtk_cell_types = ones(n_cells, dtype=int) * cell_type
        if self.debug_cell_data:
            print('vtk_cells_array', vtk_cell_array)
            print('vtk_cell_offsets', vtk_cell_offsets)
            print('vtk_cell_types', vtk_cell_types)

        return vtk_cell_array.flatten(), vtk_cell_offsets, vtk_cell_types

    def get_vtk_r_arr(self, idx):
        return self.rt_triangles[idx][0]

    def get_vtk_pnt_ip_map(self, idx):
        return self.fets_eval.get_vtk_pnt_ip_map_data(self.rt_triangles[idx][0])

    vtk_ls_values = Property(
        depends_on='sdomain.changed_structure, sdomain.+changed_geometry')

    @cached_property
    def _get_vtk_ls_values(self):
        vtk_val = []
        # TODO:define the ineraction wirh ls
        ls_fn = frompyfunc(self.sdomain.ls_fn_X, 2, 1)
        for ip_coords_X_e in self.vtk_X:
            X, Y, Z = ip_coords_X_e.T
            vtk_val.append(ls_fn(X, Y))  # TODO:3D
        return vtk_val

    def _apply_on_ip_pnts(self, fn):
        X_el = self.sdomain.elem_X_map

        # Prepare the result array of the same dimension as the result of one call to fn (for a single ip
        # - must get the first ip of the first element
        ###

        # test call to the function with single output - to get the shape of
        # the result.
        out_single = fn(
            self.ip_coords[0], X_el[0], self.dof_node_ls_values[0], self.ip_ls_values[0])
        out_grid_shape = (self.ip_coords.shape[0],) + out_single.shape
        out_grid = zeros(out_grid_shape)

        # loop over elements
        i_el = 0
        for ip_addr0, ip_addr1, X_e, node_ls in zip(self.ip_offset[:-1],
                                                    self.ip_offset[1:],
                                                    X_el,
                                                    self.dof_node_ls_values):
            ip_slice = slice(ip_addr0, ip_addr1)
            for ip_r, ip_ls in zip(self.ip_coords[ip_slice],
                                   self.ip_ls_values[ip_slice]):  # this could be dangerous when the discontinuity has more int pts than 'volume', othervise it is just overwritten in the procedure
                out_grid[i_el] = fn(ip_r, X_e, node_ls, ip_ls)
                i_el += 1
        return out_grid

    B_mtx_grid = Property(Array,
                          depends_on='sdomain.changed_structure, sdomain.+changed_geometry')

    @cached_property
    def _get_B_mtx_grid(self):
        B_mtx_grid = self._apply_on_ip_pnts(self.fets_eval.get_B_mtx)
        return B_mtx_grid

    J_det_grid = Property(Array,
                          depends_on='sdomain.changed_structure, sdomain.+changed_geometry')

    @cached_property
    def _get_J_det_grid(self):
        return self._apply_on_ip_pnts(self.fets_eval.get_J_det)

    # Integration over the discontinuity domain
    #
    def _apply_on_ip_disc(self, fn):
        X_el = self.sdomain.elem_X_map
        X_d = self.X_i

        # Prepare the result array of the same dimension as the result of one call to fn (for a single ip
        # - must get the first ip of the first element
        ###

        # test call to the function with single output - to get the shape of
        # the result.
        out_single = fn(self.ip_disc_coords[0], X_d[0], X_el[
                        0], self.dof_node_ls_values[0], self.ip_normal[0])
        out_grid_shape = (self.ip_disc_coords.shape[0],) + out_single.shape
        out_grid = zeros(out_grid_shape)

        # loop over elements
        i_el = 0
        for ip_addr0, ip_addr1, X_ed, X_el, node_ls in zip(self.ip_disc_offset[:-1],
                                                           self.ip_disc_offset[
                                                               1:],
                                                           X_d,
                                                           X_el,
                                                           self.dof_node_ls_values):
            ip_slice = slice(ip_addr0, ip_addr1)
            for ip_r, ip_norm in zip(self.ip_disc_coords[ip_slice],
                                     self.ip_normal[ip_slice]):
                out_grid[i_el] = fn(ip_r, X_ed, X_el, node_ls, ip_norm)
                i_el += 1
        return out_grid

    # Cached terms for in the integration points in the discontinuity domain.
    B_disc_grid = Property(Array,
                           depends_on='sdomain.changed_structure, sdomain.+changed_geometry')

    @cached_property
    def _get_B_disc_grid(self):
        return self._apply_on_ip_disc(self.fets_eval.get_B_disc)

    J_disc_grid = Property(Array,
                           depends_on='sdomain.changed_structure, sdomain.+changed_geometry')

    @cached_property
    def _get_J_disc_grid(self):
        return self._apply_on_ip_disc(self.fets_eval.get_J_det_disc)

    X_i = Property(
        depends_on='sdomain.changed_structure, sdomain.+changed_geometry')

    @cached_property
    def _get_X_i(self):
        '''
        Intersection points in global coords 3D
        '''
        X_i = []
        for e, r_i_e in zip(self.sdomain.elements,
                            self.sdomain.ls_intersection_r):
            X_mtx = e.get_X_mtx()
            X_i.append(self.fets_eval.get_vtk_r_glb_arr(X_mtx, r_i_e))
        return array(X_i)

    #########################################################################
    # STATE ARRAY MANAGEMENT
    #########################################################################
    state_array_size = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_state_array_size(self):
        '''
        overloading the default method
        as the number of ip differs from element to element
        '''
        n_ip = self.ip_weights.shape[0]  # number of ip on the domain
        mats_arr_size = self.fets_eval.m_arr_size
        dots_arr_size = n_ip * mats_arr_size
        return dots_arr_size

    #---------------------------------------------------
    # backup state arrays storing the values of the previous
    # --------------------------------------------------
    old_state_start_elem_grid = Array
    old_state_end_elem_grid = Array
    old_state_array = Array
    old_state_grid_ix = Tuple

    state_array = Property(Array, depends_on='sdomain.changed_structure')

    @cached_property
    def _get_state_array(self):
        '''
        overloading the default method
        state array of fets has to account for number of ip
        '''
        state_array = zeros((self.state_array_size,), dtype='float_')

        sctx = self.sdomain.domain.new_scontext()
        # Run the setup of sub-evaluator
        #
        mats_arr_size = self.fets_eval.m_arr_size
        # print 'mats_arr_size ', mats_arr_size
        # print 'self.ip_offset ', self.ip_offset
        for e_id, elem in enumerate(self.sdomain.elements):
            sctx.elem = elem
            sctx.elem_state_array = state_array[
                self.ip_offset[e_id] * mats_arr_size: self.ip_offset[(e_id + 1)] * mats_arr_size]
            # print 'elem_state_array ', sctx.elem_state_array
            self.fets_eval.setup(
                sctx, (self.ip_offset[(e_id + 1)] - self.ip_offset[e_id]))

        # Transfer the values from the old state array - if an old_array was
        # there
        if len(self.old_state_array):

            # get the start-end indices of the elements in the old grid
            old_start_arr = self.old_state_start_elem_grid[
                self.old_state_grid_ix]
            old_end_arr = self.old_state_end_elem_grid[self.old_state_grid_ix]

            # get the start-end indexes of the elements in the new grid
            new_start_arr = self.state_start_elem_grid[self.old_state_grid_ix]
            new_end_arr = self.state_end_elem_grid[self.old_state_grid_ix]

            # the elements in the new grid might get masked so that they should
            # be skipped - what should happen with their state - actually
            # a state transfer should be started.
            for new_masked, ns, ne, os, oe in zip(new_start_arr.mask,
                                                  new_start_arr, new_end_arr,
                                                  old_start_arr, old_end_arr):
                if new_masked:
                    # The element has been overloaded - the old state must be
                    # transfered to the new state - this depends on the adaptive
                    # strategy at hand. Either the new state must be reiterated
                    # for the current time once again using zero state as a
                    # start vector - or the old values could be reused as start
                    # value.
                    pass
                else:
                    # The element is present also in the changed grid - copy
                    # the state to its new place in the state array
                    state_array[ns:ne] = self.old_state_array[os:oe]

        # backup the reference to an array for the case the discretization
        # changes and transfer of state variables is reguired
        self.old_state_array = state_array
        self.old_state_start_elem_grid = self.state_start_elem_grid
        self.old_state_end_elem_grid = self.state_end_elem_grid
        self.old_state_grid_ix = self.sdomain.intg_grid_ix

        # return the new state array
        #
        return state_array

    def get_elem_state_array(self, e_id):
        '''
        used for response tracing
        @param e_id: element id
        '''
        mats_arr_size = self.fets_eval.m_arr_size
        return self.state_array[self.ip_offset[e_id] * mats_arr_size: self.ip_offset[(e_id + 1)] * mats_arr_size]

    def get_corr_pred(self, sctx, u, du, tn, tn1, F_int):

        # in order to avoid allocation of the array in every time step
        # of the computation
        k_arr = self.k_arr
        k_arr[...] = 0.0
        #k_con = zeros( ( self.fets_eval.n_e_dofs, self.fets_eval.n_e_dofs ) )
        mats_arr_size = self.fets_eval.m_arr_size

        if self.cache_geo_matrices:
            B_mtx_grid = self.B_mtx_grid
            J_det_grid = self.J_det_grid
            if self.fets_eval.nip_disc:
                B_disc_grid = self.B_disc_grid
                J_disc_grid = self.J_disc_grid

        Be_mtx_grid = None
        Je_det_grid = None

        state_array = self.state_array

        tstepper = self.sdomain.tstepper
        U = tstepper.U_k
        d_U = tstepper.d_U

        for e_id, elem, ip_addr0, ip_addr1, ip_disc_addr0, ip_disc_addr1 in zip(self.sdomain.idx_active_elems,
                                                                                self.sdomain.elements,
                                                                                self.ip_offset[
                                                                                    :-1],
                                                                                self.ip_offset[
                                                                                    1:],
                                                                                self.ip_disc_offset[
                                                                                    :-1],
                                                                                self.ip_disc_offset[1:]):
            ip_slice = slice(ip_addr0, ip_addr1)
            ip_disc_slice = slice(ip_disc_addr0, ip_disc_addr1)
            ix = elem.get_dof_map()
            sctx.elem = elem
            #sctx.elem_state_array = state_array[ e_id * e_arr_size : ( e_id + 1 ) * e_arr_size ]
            # print 'sctx.elem_state_array ', sctx.elem_state_array
            sctx.elem_state_array = state_array[self.ip_offset[
                e_id] * mats_arr_size: self.ip_offset[(e_id + 1)] * mats_arr_size]  # differs from the homogenous case
            sctx.X = elem.get_X_mtx()
            sctx.x = elem.get_x_mtx()
            if self.cache_geo_matrices:
                # differs from the homogenious case
                Be_mtx_grid = B_mtx_grid[ip_slice, ...]
                Je_det_grid = J_det_grid[ip_slice, ...]
                if self.fets_eval.nip_disc:
                    Be_disc_grid = B_disc_grid[ip_disc_slice, ...]
                    Je_disc_grid = J_disc_grid[ip_disc_slice, ...]
            sctx.ls_val = self.ip_ls_values[ip_slice]  # values of ls in ip
            f, k = self.fets_eval.get_corr_pred(sctx, U[ix_(ix)], d_U[ix_(ix)],
                                                tn, tn1,
                                                B_mtx_grid=Be_mtx_grid,
                                                J_det_grid=Je_det_grid,
                                                ip_coords=self.ip_coords[
                                                    ip_slice],
                                                ip_weights=self.ip_weights[ip_slice])

            id = [0, 2, 8, 10]
            # print 'k before \n',k[meshgrid(id,id)]
            # print 'k before \n',k
            k_arr[e_id] = k
            #k_con[:, :] = k
            F_int[ix_(ix)] += f
            # print 'f before \n',F_int
            if self.fets_eval.nip_disc:
                k_c, f_int_c = self.fets_eval.get_corr_pred_disc(sctx, U[ix_(ix)],
                                                                 B_mtx_grid=Be_disc_grid,
                                                                 J_det_grid=Je_disc_grid,
                                                                 ip_coords=self.ip_disc_coords[
                                                                     ip_disc_slice],
                                                                 ip_weights=self.ip_disc_weights[ip_disc_slice])
                k_arr[e_id] += k_c
                # print 'f_int_c ',f_int_c
                F_int[ix_(ix)] += f_int_c

            # print 'k_con ', k_con
            # print 'K__mtx', k_arr

        return SysMtxArray(mtx_arr=k_arr, dof_map_arr=self.sdomain.elem_dof_map)
