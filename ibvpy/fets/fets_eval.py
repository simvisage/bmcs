
from functools import reduce
import types

from numpy import \
    array, zeros, float_, dot, hstack, arange, argmin, broadcast_arrays, c_
from scipy.linalg import \
    det
from scipy.spatial.distance import \
    cdist
from traits.api import \
    Array, Bool, Float, provides, \
    Instance, Int, Trait, List, Any, \
    Delegate, Property, cached_property, Dict, \
    Type
from traitsui.api import \
    View, Item, Group

from ibvpy.core.i_tstepper_eval import \
    ITStepperEval
from ibvpy.core.rtrace_eval import \
    RTraceEval
from ibvpy.core.tstepper_eval import \
    TStepperEval
from ibvpy.dots.dots_eval import \
    DOTSEval
import numpy as np
from tvtk.tvtk_classes import tvtk_helper

from .i_fets_eval import IFETSEval


#-------------------------------------------------------------------
# Numpy extension
#-------------------------------------------------------------------
def oriented_3d_array(arr, axis):
    '''In order to use the indices as spatial locators
    the array of gauss points is augmented with newaxes into 3D
    so that the carthesian broadcasting can be done.

    There is only the expand_dim function available in numpy.
    Here we want to  put the supplied array in 3d space along
    the axis  
    '''
    shape = [None, None, None]
    shape[axis] = slice(None)
    _arr = array(arr, dtype='float_')
    return _arr[tuple(shape)]

#-------------------------------------------------------------------
# FETSEval - general implementation of the fe-numerical quadrature
#-------------------------------------------------------------------


@provides(IFETSEval)
class FETSEval(TStepperEval):

    dots_class = Type(DOTSEval)

    dof_r = Array('float_',
                  desc='Local coordinates of nodes included in the field ansatz')

    geo_r = Array('float_',
                  desc='Local coordinates of nodes included in the geometry ansatz')

    n_nodal_dofs = Int(desc='Number of nodal degrees of freedom')

    id_number = Int

    #-------------------------------------------------------------------------
    # Derived info about the finite element formulation
    #-------------------------------------------------------------------------
    n_dof_r = Int

    def _n_dof_r_default(self):
        return len(self.dof_r)

    n_geo_r = Int

    def _n_geo_r_default(self):
        return len(self.geo_r)

    #-------------------------------------------------------------------------
    # Field visualization
    #-------------------------------------------------------------------------

    vtk_r = Array(
        Float, desc='Local coordinates of nodes included in the field visualization')

    vtk_cell_types = Any(
        desc='Tuple of vtk cell types in the same order as they are specified in the vtk_cells list')

    vtk_cells = List(
        desc='List of maps of nodes constituting the vtk cells covering the single element')

    vtk_cell_class = Property

    def _get_vtk_cell_class(self):
        return tvtk_helper.get_class(self.vtk_cell_type)

    # Distinguish the type of base geometric entity to be used for
    # the visualization of the results.
    #
    # field_entity_type = Enum('vertex','line','triangle','quad','tetra','hexa')

    vtk_node_cell_data = Property(depends_on='vtk_cells, vtk_cell_types')

    @cached_property
    def _get_vtk_node_cell_data(self):

        n_cells = len(self.vtk_cells)

        # check if vtk_cell_types is a list, if not make one
        if isinstance(self.vtk_cell_types, str):
            cell_classes = [self.vtk_cell_types for i in range(n_cells)]
        else:
            cell_classes = self.vtk_cell_types

        cell_types = []
        for cell_str in cell_classes:
            cell_class = tvtk_helper.get_class(cell_str)
            cell_types.append(cell_class().cell_type)

        if isinstance(self.vtk_cells[0], int):
            # just a single cell defined
            return (array([0, ], dtype=int),
                    array(self.vtk_cells.shape[0], dtype=int),
                    array(self.vtk_cells, dtype=int),
                    cell_types)

        offset_list = []
        length_list = []
        cell_list = []
        vtk_offset = 0
        for cell in self.vtk_cells:
            cell_len = len(cell)
            cell_list += cell
            length_list.append(cell_len)
            offset_list.append(vtk_offset)
            vtk_offset += cell_len + 1

        return (array(offset_list, dtype=int),
                array(length_list, dtype=int),
                array(cell_list, dtype=int),
                array(cell_types, dtype=int))

    vtk_ip_cell_data = Property(depends_on='vtk_cells, vtk_cell_types')

    @cached_property
    def _get_vtk_ip_cell_data(self):

        n_ip_pnts = self.ip_coords.shape[0]

        cell_types = array([(tvtk_helper.get_class('PolyVertex')()).cell_type])

        return (array([0, ], dtype=int),
                array([n_ip_pnts], dtype=int),
                arange(n_ip_pnts),
                cell_types)

    n_vtk_r = Property(Int, depends_on='vtk_r')

    @cached_property
    def _get_n_vtk_r(self):
        return self.vtk_r.shape[0]

    n_vtk_cells = Property(Int, depends_on='field_faces')

    @cached_property
    def _get_n_vtk_cells(self):
        return self.field_faces.shape[0]

    vtk_pnt_ip_map = Property(Array(Int))

    @cached_property
    def _get_vtk_pnt_ip_map(self):
        return self.get_vtk_pnt_ip_map_data(self.vtk_r)

    def adjust_spatial_context_for_point(self, sctx):
        '''
        Method gets called prior to the evaluation at the material point level.

        The method can be used for dimensionally reduced evaluators.
        This is FETS specific and should be moved there.
        However, the RTraceEval is not distinguished at the moment, therefore
        it is here - move!!!.   
        '''
        sctx.X_reg = sctx.X

    def get_vtk_pnt_ip_map_data(self, vtk_r):
        '''
        mapping of the visualization point to the integration points
        according to mutual proximity in the local coordinates
        '''
        vtk_pt_arr = zeros((1, 3), dtype='float_')
        ip_map = zeros(vtk_r.shape[0], dtype='int_')
        for i, vtk_pt in enumerate(vtk_r):
            vtk_pt_arr[0, self.dim_slice] = vtk_pt[self.dim_slice]
            # get the nearest ip_coord
            ip_map[i] = argmin(cdist(vtk_pt_arr, self.ip_coords))
        return array(ip_map)

    #-------------------------------------------------------------------------
    # NUMERICAL INTEGRATION
    #-------------------------------------------------------------------------
    #
    # The integration grid is constructed using broadcasting
    # provided by numpy. This provides a loopless implementation
    # with well defined ordering of gauss points and further the
    # slicing of gauss points using the numpy indexing slices.
    #
    # The old loop-based expansion is preserved below for reference.
    #
    gp_r_grid = Property(depends_on='ngp_r,ngp_s,ngp_t')

    @cached_property
    def _get_gp_r_grid(self):
        '''Return a tuple of three arrays for X, Y, Z coordinates of the
        gauss points within the element.
        '''
        # get the oriented arrays for each direction prepared for broadcasting
        #
        gp_coords = [oriented_3d_array(self._GP_COORDS[ngp], dim)
                     for dim, ngp
                     in enumerate(self.n_gp_list)]

        # broadcast the values to construct all combinations of all gauss point
        # coordinates.
        #
        x, y, z = broadcast_arrays(*gp_coords)
        return x, y, z

    gp_w_grid = Property(depends_on='ngp_r,ngp_s,ngp_t')

    @cached_property
    def _get_gp_w_grid(self):
        '''In analogy to the above, get the grid of gauss weights in 3D.
        '''
        # get the oriented arrays for each direction prepared for broadcasting
        #
        gp_w = [oriented_3d_array(self._GP_WEIGHTS[ngp], dim) for dim, ngp
                in enumerate(self.n_gp_list)]

        # broadcast the values to construct all combinations of all gauss point
        # coordinates.
        #
        w = reduce(lambda x, y: x * y, gp_w)
        return w

    ip_coords = Property(depends_on='ngp_r,ngp_s,ngp_t')

    def _get_ip_coords(self):
        '''Generate the flat array of ip_coords used for integration.
        '''
        x, y, z = self.gp_r_grid
        return c_[x.flatten(), y.flatten(), z.flatten()]

    r_m = Property

    def _get_r_m(self):
        return self.ip_coords

    ip_coords_grid = Property(depends_on='ngp_r,ngp_s,ngp_t')

    def _get_ip_coords_grid(self):
        '''Generate the grid of ip_coords
        '''
        return c_[self.gp_r_grid]

    ip_weights = Property(depends_on='ngp_r,ngp_s,ngp_t')

    w_m = Property

    def _get_w_m(self):
        return self.ip_weights

    n_m = Property

    def _get_n_m(self):
        return len(self.ip_weights)

    def _get_ip_weights(self):
        '''Generate the flat array of ip_coords used for integration.
        '''
        w = self.gp_w_grid
        return w.flatten()

    ip_weights_grid = Property(depends_on='ngp_r,ngp_s,ngp_t')

    def _get_ip_weights_grid(self):
        '''Generate the flat array of ip_coords used for integration.
        '''
        w = self.gp_w_grid
        return w

    def get_ip_scheme(self, *params):
        return (self.ip_coords, self.ip_weights)

    n_gp = Property(depends_on='ngp_r,ngp_s,ngp_t')

    @cached_property
    def _get_n_gp(self):
        nr = max(1, self.ngp_r)
        ns = max(1, self.ngp_s)
        nt = max(1, self.ngp_t)
        return nr * ns * nt

    n_gp_list = Property(depends_on='ngp_r,ngp_s,ngp_r')

    @cached_property
    def _get_n_gp_list(self):
        nr = self.ngp_r
        ns = self.ngp_s
        nt = self.ngp_t
        return [nr, ns, nt]
    #-------------------------------------------------------------------------
    # SUBSPACE INTEGRATION
    #-------------------------------------------------------------------------
    #
    # Get the integration scheme for a subspace specified using the slicing indexes.
    #

    def get_sliced_ip_scheme(self, ip_idx_list):

        minmax = {0: min, -1: max}
        w = []
        r = []
        ix = []
        for dim_idx, ip_idx in enumerate(ip_idx_list):
            if isinstance(ip_idx, int):
                w.append(1.0)
                r.append(minmax[ip_idx](self.dof_r[:, dim_idx]))
            elif isinstance(ip_idx, slice):
                # if there is a slice - put the array in the corresponding dimension
                # - only the full slide - i.e. slice(None,None,None)
                #   is allowed
                n_gp = self.n_gp_list[dim_idx]
                w.append(oriented_3d_array(self._GP_WEIGHTS[n_gp], dim_idx))
                r.append(oriented_3d_array(self._GP_COORDS[n_gp], dim_idx))
                ix.append(dim_idx)
        r_grid = broadcast_arrays(*r)
        r_c = c_[tuple([r.flatten() for r in r_grid])]
        w_grid = reduce(lambda x, y: x * y, w)
        if isinstance(w_grid, float):
            w_grid = array([w_grid], dtype='float_')
        else:
            w_grid = w_grid.flatten()
        return r_c, w_grid, ix

    #-------------------------------------------------------------------------
    # FIELD TRACING / VISUALIZATION
    #-------------------------------------------------------------------------
    # The user-specified fv_loc_coords list gets transform to an internal
    # array representation
    #
    vtk_r_arr = Property(depends_on='vtk_r')

    @cached_property
    def _get_vtk_r_arr(self):
        if len(self.vtk_r) == 0:
            raise ValueError(
                'Cannot generate plot, no vtk_r specified in fets_eval')
        return array(self.vtk_r)

    def get_vtk_r_glb_arr(self, X_mtx, r_mtx=None):
        '''
        Get an array with global coordinates of the element decomposition.

        If the local_point_list is non-empty then use it instead of the one supplied 
        by the element specification. This is useful for augmented specification of RTraceEval 
        evaluations with a specific profile of a field variable to be traced.
        '''
        if self.dim_slice:
            X_mtx = X_mtx[:, self.dim_slice]

        if r_mtx == None:
            r_mtx = self.vtk_r_arr

        # TODO - efficiency in the extraction of the global coordinates. Is broadcasting
        # a possibility - we only need to augment the matrix with zero coordinates in the
        # in the unhandled dimensions.
        #
        X3D = array([dot(self.get_N_geo_mtx(r_pnt), X_mtx)[0, :]
                     for r_pnt in r_mtx])
        n_dims = r_mtx.shape[1]
        n_add = 3 - n_dims
        if n_add > 0:
            X3D = hstack([X3D,
                          zeros([r_mtx.shape[0], n_add], dtype='float_')])
        return X3D

    def get_X_pnt(self, sctx):
        '''
        Get the global coordinates for the specified local coordinats r_pnt
        @param r_pnt: local coordinates
        '''
        r_pnt = sctx.r_pnt
        X_mtx = sctx.X

        return np.einsum('', self.Nr_i_geo, X_mtx)
        return dot(self.get_N_geo(r_pnt), X_mtx)

    def get_x_pnt(self, sctx):
        '''
        Get the global coordinates for the specified local coordinats r_pnt
        @param r_pnt: local coordinates
        '''
        r_pnt = sctx.r_pnt
        x_mtx = sctx.x

        # TODO - efficiency in the extraction of the global coordinates. Is broadcasting
        # a possibility - we only need to augment the matrix with zero coordinates in the
        # in the unhandled dimensions.
        #
        return dot(self.get_N_geo(r_pnt), x_mtx)

    def map_r2X(self, r_pnt, X_mtx):
        '''
        Map the local coords to global
        @param r_pnt: local coords
        @param X_mtx: matrix of the global coords of geo nodes
        '''
        # print "mapping ",dot( self.get_N_geo_mtx(r_pnt)[0], X_mtx )," ",
        # r_pnt
        return dot(self.get_N_geo(r_pnt), X_mtx)

    # Number of element DOFs
    #
    n_e_dofs = Int

    # Dimensionality
    dim_slice = None

    # Parameters for the time-loop
    #
    def new_cntl_var(self):
        return zeros(self.n_e_dofs, float_)

    def new_resp_var(self):
        return zeros(self.n_e_dofs, float_)

    def get_state_array_size(self):
        r_range = max(1, self.ngp_r)
        s_range = max(1, self.ngp_s)
        t_range = max(1, self.ngp_t)
        return self.m_arr_size * r_range * s_range * t_range

    m_arr_size = Property()

    @cached_property
    def _get_m_arr_size(self):
        return self.get_mp_state_array_size(None)

    ngp_r = Int(0, label='Number of Gauss points in r-direction')
    ngp_s = Int(0, label='Number of Gauss points in s-direction')
    ngp_t = Int(0, label='Number of Gauss points in t-direction')

    #-------------------------------------------------------------------
    # Overloadable methods
    #-------------------------------------------------------------------
    def get_corr_pred(self, sctx, u, du, tn, tn1,
                      eps_avg=None,
                      B_mtx_grid=None,
                      J_det_grid=None,
                      ip_coords=None,
                      ip_weights=None):
        '''
        Corrector and predictor evaluation.

        @param u current element displacement vector
        '''
        u_avg = eps_avg  # temporary
        if J_det_grid == None or B_mtx_grid == None:
            #            if self.dim_slice:
            #                X_mtx = sctx.X[:, self.dim_slice]
            #            else:
            X_mtx = sctx.X

        show_comparison = True
        if ip_coords == None:
            ip_coords = self.ip_coords
            show_comparison = False
        if ip_weights == None:
            ip_weights = self.ip_weights

        # Use for Jacobi Transformation

        n_e_dofs = self.n_e_dofs
        K = zeros((n_e_dofs, n_e_dofs))
        F = zeros(n_e_dofs)
        sctx.fets_eval = self

        ip = 0      # use enumerate

        # Element formulation-specific adjustment of the spatial context
        # for the material level (needed for combination of regularized
        # material models with elements of lower dimensions.
        #
        self.adjust_spatial_context_for_point(sctx)

        # Numerical quadrature loop
        #
        for r_pnt, wt in zip(ip_coords, ip_weights):

            sctx.r_pnt = r_pnt
            if J_det_grid == None:
                J_det = self._get_J_det(r_pnt, X_mtx)
            else:
                J_det = J_det_grid[ip, ...]
            if B_mtx_grid == None:
                B_mtx = self.get_B_mtx(r_pnt, X_mtx)
            else:
                B_mtx = B_mtx_grid[ip, ...]

            # Map displacements to strains
            #
            eps_mtx = dot(B_mtx, u)
            d_eps_mtx = dot(B_mtx, du)

            # Set the state array slice into the spatial context
            #
            sctx.mats_state_array = sctx.elem_state_array[
                ip * self.m_arr_size: (ip + 1) * self.m_arr_size]

            # Evaluate the corrector and predictor for the current iteration
            #
            if u_avg != None:
                eps_avg = dot(B_mtx, u_avg)
                sig_mtx, D_mtx = self.get_mtrl_corr_pred(
                    sctx, eps_mtx, d_eps_mtx, tn, tn1, eps_avg)
            else:
                sig_mtx, D_mtx = self.get_mtrl_corr_pred(
                    sctx, eps_mtx, d_eps_mtx, tn, tn1)

            # Evaluate the element stiffness matrix
            #
            k = dot(B_mtx.T, dot(D_mtx, B_mtx))
            k *= (wt * J_det)
            K += k

            # Evaluate the internal force vector
            #
            f = dot(B_mtx.T, sig_mtx)
            f *= (wt * J_det)
            F += f

            ip += 1

        return F, K

    #-------------------------------------------------------------------
    # Standard evaluation methods
    #-------------------------------------------------------------------
    def get_J_mtx(self, r_pnt, X_mtx):
        dNr_geo_mtx = self.get_dNr_geo_mtx(r_pnt)
        return dot(dNr_geo_mtx, X_mtx)

    #-------------------------------------------------------------------
    # Required methods
    #-------------------------------------------------------------------

    def get_N_geo_mtx(self, r_pnt):
        raise NotImplementedError

    def get_dNr_geo_mtx(self, r_pnt):
        raise NotImplementedError

    def get_N_mtx(self, r_pnt):
        raise NotImplementedError

    def get_B_mtx(self, r_pnt, X_mtx):
        '''
        Get the matrix for kinematic mapping between displacements and strains.
        @param r local position within the element.
        @param X nodal coordinates of the element.

        @TODO[jakub] generalize
        '''
        raise NotImplementedError

    def get_mtrl_corr_pred(self, sctx, eps_eng, d_eps_eng, tn, tn1, eps_avg=None):
        if self.mats_eval.initial_strain:
            X_pnt = self.get_X_pnt(sctx)
            x_pnt = self.get_x_pnt(sctx)
            eps_ini_mtx = self.mats_eval.initial_strain(X_pnt, x_pnt)
            eps0_eng = self.mats_eval.map_eps_mtx_to_eng(eps_ini_mtx)
            eps_eng -= eps0_eng
        if eps_avg != None:
            sig_mtx, D_mtx = self.mats_eval.get_corr_pred(
                sctx, eps_eng, d_eps_eng, tn, tn1, eps_avg)
        else:
            sig_mtx, D_mtx = self.mats_eval.get_corr_pred(
                sctx, eps_eng, d_eps_eng, tn, tn1,)
        return sig_mtx, D_mtx

    #-------------------------------------------------------------------
    # Private methods
    #-------------------------------------------------------------------

    def get_J_det(self, r_pnt, X_mtx):
        return array(self._get_J_det(r_pnt, X_mtx), dtype='float_')

    def _get_J_det(self, r_pnt3d, X_mtx):
        if self.dim_slice:
            r_pnt = r_pnt3d[self.dim_slice]
        return det(self.get_J_mtx(r_pnt, X_mtx))

    # if no gauss point is defined in one direction (e.g. for ngp_t=0 for a 2D-problem)
    # then the default value for ngp_t=0 is used and a weighting coefficient of value 1.
    # In 'get_gp' the maximum of npg and 1 is used as range in the loop which leads to
    # a simple multiplication with 1 for that direction. The coordinate of the gauss point
    # in this direction is set to zero ('gp' : [0.]).

    # The index of the entry corresponds to the order of the polynomial
    #
    _GP_COORDS = [[0.],  # for polynomial order  = 0
                  [0.],  # for polynomial order  = 1
                  # por polynomial order = 2
                  [-0.57735026918962584, 0.57735026918962584],
                  # for polynomial order = 3
                  [-0.7745966692414834, 0., 0.7745966692414834],
                  # for polynomial order = 4
                  [-0.861136311594053, -0.339981043584856,
                   0.339981043584856, 0.861136311594053]
                  ]

    _GP_WEIGHTS = [[1.],
                   [2.],
                   [1., 1.],
                   [0.55555555555555558, 0.88888888888888884,
                       0.55555555555555558],
                   [0.347854845137454, 0.652145154862546,
                       0.652145154862546, 0.347854845137454]
                   ]

    #-------------------------------------------------------------------------
    # Epsilon as an engineering value
    #-------------------------------------------------------------------------
    def get_eps0_eng(self, sctx, u):
        '''Get epsilon without the initial strain
        '''
        if self.mats_eval.initial_strain:
            X_pnt = self.get_X_pnt(sctx)
            x_pnt = self.get_x_pnt(sctx)
            eps0_mtx = self.mats_eval.initial_strain(X_pnt, x_pnt)
            return self.mats_eval.map_eps_mtx_to_eng(eps0_mtx)
        else:
            return None

    def get_eps_eng(self, sctx, u):
        X_mtx = sctx.X
        r_pnt = sctx.loc
        B_mtx = self.get_B_mtx(r_pnt, X_mtx)
        eps_eng = dot(B_mtx, u)
        return eps_eng

    def get_eps1t_eng(self, sctx, u):
        '''Get epsilon without the initial strain
        '''
        eps_eng = self.get_eps_eng(sctx, u)
        eps0_eng = self.get_eps0_eng(sctx, u)
        if eps0_eng != None:
            eps_eng -= eps0_eng
        return eps_eng

    #-------------------------------------------------------------------------
    # Epsilon as a tensorial value - used for visualization in mayavi
    #-------------------------------------------------------------------------
    def get_eps0_mtx33(self, sctx, u):
        '''Get epsilon without the initial strain
        '''
        eps0_mtx33 = zeros((3, 3), dtype='float_')
        if self.mats_eval.initial_strain:
            X_pnt = self.get_X_pnt(sctx)
            x_pnt = self.get_x_pnt(sctx)
            eps0_mtx33[self.dim_slice, self.dim_slice] = self.mats_eval.initial_strain(
                X_pnt, x_pnt)
        return eps0_mtx33

    def get_eps_mtx33(self, sctx, u):
        eps_eng = self.get_eps_eng(sctx, u)
        eps_mtx33 = zeros((3, 3), dtype='float_')
        eps_mtx33[self.dim_slice, self.dim_slice] = self.mats_eval.map_eps_eng_to_mtx(
            eps_eng)
        return eps_mtx33

    def get_eps1t_mtx33(self, sctx, u):
        '''Get epsilon without the initial strain
        '''
        eps_mtx33 = self.get_eps_mtx33(sctx, u)
        eps0_mtx33 = self.get_eps0_mtx33(sctx, u)
        return eps_mtx33 - eps0_mtx33

    #-------------------------------------------------------------------------
    # Displacement in a n arbitrary point within an element
    #-------------------------------------------------------------------------
    def map_eps(self, sctx, u):
        X_mtx = sctx.X
        r_pnt = sctx.loc
        B_mtx = self.get_B_mtx(r_pnt, X_mtx)
        eps = dot(B_mtx, u)
        return eps

    def get_u(self, sctx, u):
        N_mtx = self.get_N_mtx(sctx.loc)
        return dot(N_mtx, u)

    debug_on = Bool(False)

    def _debug_rte_dict(self):
        '''
        RTraceEval dictionary with field variables used to verify the element implementation
        '''
        if self.debug_on:
            return {'Ngeo_mtx': RTraceEvalElemFieldVar(eval=lambda sctx, u: self.get_N_geo_mtx(sctx.loc),
                                                       ts=self),
                    'N_mtx': RTraceEvalElemFieldVar(eval=lambda sctx, u: self.get_N_mtx(sctx.loc)[0],
                                                    ts=self),
                    'B_mtx0': RTraceEvalElemFieldVar(eval=lambda sctx, u: self.get_B_mtx(sctx.loc, sctx.X)[0],
                                                     ts=self),
                    'B_mtx1': RTraceEvalElemFieldVar(eval=lambda sctx, u: self.get_B_mtx(sctx.loc, sctx.X)[1],
                                                     ts=self),
                    'B_mtx2': RTraceEvalElemFieldVar(eval=lambda sctx, u: self.get_B_mtx(sctx.loc, sctx.X)[2],
                                                     ts=self),
                    'J_det': RTraceEvalElemFieldVar(eval=lambda sctx, u:
                                                    array(
                                                        [det(self.get_J_mtx(sctx.loc, sctx.X))]),
                                                    ts=self)}
        else:
            return {}

    # List of mats that are to be chained
    #

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        '''
        RTraceEval dictionary with standard field variables.
        '''
        rte_dict = self._debug_rte_dict()
        for key, v_eval in list(self.mats_eval.rte_dict.items()):

            # add the eval into the loop.
            #
            rte_dict[key] = RTraceEvalElemFieldVar(name=key,
                                                   u_mapping=self.get_eps1t_eng,
                                                   eval=v_eval)

        rte_dict.update({'eps_app': RTraceEvalElemFieldVar(eval=self.get_eps_mtx33),
                         'eps0_app': RTraceEvalElemFieldVar(eval=self.get_eps0_mtx33),
                         'eps1t_app': RTraceEvalElemFieldVar(eval=self.get_eps1t_mtx33),
                         'u': RTraceEvalElemFieldVar(eval=self.get_u)})

        return rte_dict

    traits_view = View(
        Group(
            Item('n_e_dofs'),
            Item('n_nodal_dofs'),
            label='Numerical parameters'
        ),
        Group(
            #                              Item( 'dof_r' ),
            #                              Item( 'geo_r' ),
            Item('vtk_r'),
            Item('vtk_cells'),
            Item('vtk_cell_types'),
            label='Visualization parameters'
        ),
        #                         Item('rte_dict'),
        resizable=True,
        scrollable=True,
        width=0.2,
        height=0.4
    )


class RTraceIntegEvalElemFieldVar(RTraceEval):

    integral = True
    # To be specialized for element level
    #

    def __call__(self, sctx, u, B_mtx_grid=None, J_det_grid=None):
        if J_det_grid == None or B_mtx_grid == None:
            X_mtx = sctx.X
#            if self.dim_slice:
#                X_mtx = sctx.X[:, self.dim_slice]
#            else:

        show_comparison = True
        if ip_coords == None:
            ip_coords = self.ip_coords
            show_comparison = False
        if ip_weights == None:
            ip_weights = self.ip_weights

        # Use for Jacobi Transformation

        F = 0.0
        sctx.fets_eval = self

        ip = 0      # use enumerate

        for r_pnt, wt in zip(ip_coords, ip_weights):
            sctx.r_pnt = r_pnt
            if J_det_grid == None:
                J_det = self._get_J_det(r_pnt, X_mtx)
            else:
                J_det = J_det_grid[ip, ...]
            if B_mtx_grid == None:
                B_mtx = self.get_B_mtx(r_pnt, X_mtx)
            else:
                B_mtx = B_mtx_grid[ip, ...]
            eps_mtx = dot(B_mtx, u)
            sctx.mats_state_array = sctx.elem_state_array[
                ip * self.m_arr_size: (ip + 1) * self.m_arr_size]
            f = self.eval(sctx, eps_mtx)
            f *= (wt * J_det)
            F += f

        return F


class RTraceEvalElemFieldVar(RTraceEval):

    # To be specialized for element level
    #
    field_entity_type = Delegate('ts')
    vtk_r_arr = Delegate('ts')
    get_vtk_r_glb_arr = Delegate('ts')
    field_vertexes = Delegate('ts')
    field_lines = Delegate('ts')
    field_faces = Delegate('ts')
    field_volumes = Delegate('ts')
    n_vtk_cells = Delegate('ts')
    vtk_cell_data = Delegate('ts')
