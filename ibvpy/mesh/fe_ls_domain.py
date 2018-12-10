
from numpy import \
    array, zeros, repeat, arange, \
    hstack, sort, where, unique, copy, frompyfunc, sign, fabs as nfabs, \
    logical_and, argwhere, fabs, ma, logical_or, \
    zeros_like
from scipy.optimize import \
    brentq
from traits.api import \
    Instance, Property, cached_property, \
    provides, Float, \
    Callable, Str, Enum, on_trait_change, Any, \
    Event

from ibvpy.dots.xdots_eval import XDOTSEval
from ibvpy.fets.fets_eval import FETSEval
from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.mesh.i_fe_uniform_domain import IFEUniformDomain
from ibvpy.rtrace.rt_domain import RTraceDomain

from .fe_grid import MElem
from .fe_subdomain import FESubDomain


#--------------------------------------------------------------------------
# Summation procedure on an n-dimensional grid
#--------------------------------------------------------------------------
def edge_sum(b_arr):
    '''Perform summation of values in neighboring nodes
    along edges in each dimension.

    The original array [n,n,n] is reduced to [n-1,n-1,n-1] 
    '''
    n_dims = len(b_arr.shape)

    # construct the slice template - all indices are running through
    s_list = [
        [slice(None)
         for i in range(n_dims)
         ]
        for j in range(n_dims)
    ]

    # construct the left and right index
    # sl = :-1   - from first to second last
    # sr = 1:    - from second to last
    sl, sr = slice(None, -1), slice(1, None)

    # construct a slice of the type
    # b_arr[ :, :-1, : ] + b_arr[ :, 1:, : ]
    # performing the neighbor summation in all dimensions
    #
    for i, s in enumerate(s_list):
        slice_l, slice_r = copy(s), copy(s)
        slice_l[i] = sl
        slice_r[i] = sr
        b_arr = b_arr[tuple(slice_l)] + b_arr[tuple(slice_r)]

    return b_arr


def get_intersect_pt(fn, args):
    try:
        return brentq(fn, -1, 1, args=args)
    except ValueError:
        return


@provides(IFEUniformDomain)
class FELSDomain(FESubDomain):
    '''Domain defined using a level set function within an FEGrid.
    '''

    # Distinguish which part of the level set to take
    #
    ls_side_tag = Enum('both', 'pos', 'neg')

    rt_tol = Float(0.0001)

    # specialized label
    _tree_label = Str('extension domain')

    # container domain (why is this needed?)
    _domain = Instance('ibvpy.mesh.fe_domain.FEDomain')
    domain = Property

    def _set_domain(self, value):
        'reset the domain of this domain'
        if self._domain:
            # unregister in the old domain
            raise NotImplementedError(
                'FESubDomain cannot be relinked to another FEDomain')

        self._domain = value
        # register in the domain as a subdomain
        self._domain.xdomains.append(self)
        self._domain._append_in_series(self)

    def _get_domain(self):
        return self._domain

    #-----------------------------------------------------------------
    # Associated time stepper
    #-----------------------------------------------------------------

    # element level
    fets_eval = Instance(FETSEval)

    # domain level
    dots = Property

    @cached_property
    def _get_dots(self):
        '''Construct and return a new instance of domain
        time stepper.
        '''
        return XDOTSEval(sdomain=self)

    #-----------------------------------------------------------------
    # Response tracer domain
    #-----------------------------------------------------------------

    rt_bg_domain = Property(depends_on='+changed_structure,+changed_geometry')

    @cached_property
    def _get_rt_bg_domain(self):
        return RTraceDomain(sd=self)

    def redraw(self):
        self.rt_bg_domain.redraw()

    #-----------------------------------------------------------------
    # Boundary level set - define the boundary of the subdomain
    #-----------------------------------------------------------------
    # Specify the boundary of the subdomain
    # The elements outside of the domain are
    # not included in the integration and are
    # not deactivated from the original mesh.
    #
    bls_function = Callable(ls_changed=True)

    def _bls_function_default(self):
        def unity(*x):
            # by default the level set function is positive in the whole domain
            # - the level set is valid in the whole domain
            return sign(fabs(x[0]))
        return unity

    #-----------------------------------------------------------------
    # Level set separating the domain into the positive and negative parts
    #-----------------------------------------------------------------

    # current slice instance
    ls_function = Callable(ls_changed=True)

    # parent grid - should be included in the hierarchical fe_domain structure provided
    # by the FEParentDomain.
    #
    fe_grid = Any

    #---------------------------------------------------------------------
    # Grids with the values of the level sets
    #---------------------------------------------------------------------
    bls_edge_sum_grid = Property(depends_on='+shape_changed, bls_function')

    @cached_property
    def _get_bls_edge_sum_grid(self):

        X_grid = self.fe_grid.dof_vertex_X_grid

        # evaluate the boundary operator - with a result boolean array
        # in the corner nodes.
        #
        b_vertex_grid = sign(self.bls_function(*X_grid))

        return edge_sum(b_vertex_grid)

    ls_edge_sum_grid = Property(depends_on='+shape_changed, ls_function')

    @cached_property
    def _get_ls_edge_sum_grid(self):

        X_grid = self.fe_grid.dof_vertex_X_grid

        # find the indices of the elems

        ls_function = self.ls_function

        n_dims = X_grid.shape[0]

        vect_fn = frompyfunc(ls_function, n_dims, 1)
        ls_vertex_grid = sign(vect_fn(*X_grid))

        return array(edge_sum(ls_vertex_grid), dtype=float)

    changed_structure = Event

    @on_trait_change('shape_changed,+ls_changed,idx_mask_changed')
    def _set_changed_structure(self):
        self.changed_structure = True

    #----------------------------------------------------------------------
    # Mask arrays - True means the element is masked
    #----------------------------------------------------------------------

    ls_mask = Property(depends_on='+shape_changed,+ls_changed')

    @cached_property
    def _get_ls_mask(self):

        # make the intersection
        #
        # Three conditions must be fulfilled:
        # 1) The absolute value of the level summation on the element is less than
        # the number of the corner nodes
        #
        # 2) At least one value of the boundary level set must be positive within the element
        # i.e. the sum within the element is greater than -(number of corner nodes)
        #
        # 3) The element must be valid also in the parent grid
        #
        # True - are the masked elements, False are the active elements
        #
        ls_mask = logical_and(nfabs(self.ls_edge_sum_grid) < 4,
                              self.bls_edge_sum_grid > -4) == False

        if isinstance(self.fe_grid, FEGrid):
            n_grid_elems = self.fe_grid.n_grid_elems
            parent_ls_mask = repeat(
                False, n_grid_elems).reshape(self.fe_grid.shape)
        else:
            parent_ls_mask = self.fe_grid.ls_mask

        return logical_or(parent_ls_mask, ls_mask)

    #------------------------------------------------------------------------
    # Mask used for hiding elements in a child grid
    #------------------------------------------------------------------------
    # Extra state trait indicating that some elements have been deleted.
    #
    idx_mask_changed = Event

    # @todo define shape attribute
    idx_mask = Property(depends_on='+shape_changed,+ls_changed')

    @cached_property
    def _get_idx_mask(self):
        self.idx_mask_changed = True
        idx_mask = zeros_like(self.fe_grid.ls_mask)
        return idx_mask  # return a boolean array with False value

    xelems_mask = Property(
        depends_on='+shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_xelems_mask(self):
        '''Get elements that have been deactivated/enriched by a child.
        '''
        return logical_and(self.idx_mask == True, self.ls_mask == False)

    def deactivate_intg_elems_in_parent(self):
        '''Deactivate the elements integrated within this domain in the parent grid. 
        '''
        # deactivate the elements in the parent
        #
        if isinstance(self.fe_grid, FEGrid):
            for e in self.ls_elems:
                self.fe_grid.deactivate(e)
        elif isinstance(self.fe_grid, FELSDomain):
            # get elements active in this domain
            # deactivate them in the parent array by putting them into its mask
            self.fe_grid.idx_mask[self.intg_grid_ix] = True
            self.fe_grid.idx_mask_changed = True

    #-------------------------------------------------------------------------
    # Mask hiding elements from the integration
    #-------------------------------------------------------------------------
    # There are three views to each mesh
    # 1) the view of the integrator (time stepper)
    # 2) the view of refinement/enrichment domain reusing some information
    #    of this grid
    intg_mask = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_intg_mask(self):
        intg_mask = logical_or(self.ls_mask, self.idx_mask)
        return intg_mask

    # Inverse of the intg_mask - intg_profile
    #
    intg_profile = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_intg_profile(self):
        return self.intg_mask == False

    #----------------------------------------------------------------------
    # Array of grid indices extracting the active elements from the grids
    #----------------------------------------------------------------------
    ls_grid_ix = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_ls_grid_ix(self):
        '''Indices of the level set elements within the grid
        '''
        # get the indexes of the enriched elements
        #
        grid_ix = argwhere(self.ls_mask == False)
        ilist = [grid_ix[:, i] for i in range(grid_ix.shape[1])]
        return tuple(ilist)

    intg_grid_ix = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_intg_grid_ix(self):
        '''Indices of the integration elements within the grid
        '''
        # get the indexes of the enriched elements
        #
        grid_ix = argwhere(self.intg_mask == False)
        ilist = [grid_ix[:, i] for i in range(grid_ix.shape[1])]
        return tuple(ilist)

    xelems_grid_ix = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_xelems_grid_ix(self):
        '''Get the grid_indices of the enriched elements within the grid
        '''
        # get the indexes of the enriched elements
        #
        grid_ix = argwhere(self.xelems_mask)

        # Convert the index array to a list of tuples to extract the element numbers
        # from the ls_ielem_grid
        #
        ilist = [grid_ix[:, i] for i in range(grid_ix.shape[1])]
        return tuple(ilist)

    #-------------------------------------------------------------------------
    # Masked element grids
    #-------------------------------------------------------------------------
    ls_elem_grid = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_ls_elem_grid(self):
        '''Hide only the elements not included in the level set.
        '''
        return ma.masked_array(self.fe_grid.ls_elem_grid, self.ls_mask)

    intg_elem_grid = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_intg_elem_grid(self):
        '''Hide elements not included in the level set and those deleted explicitly by a child grid..
        '''
        return ma.masked_array(self.fe_grid.ls_elem_grid, self.intg_mask)

    ls_ielem_grid = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_ls_ielem_grid(self):
        '''Enumeration of the active elements within the domain.
        the element data (dof_map, geo_X_map, geo_x_map) using the grid index
        '''
        enum_elems = arange(self.ls_grid_ix[0].shape[0])
        elem_grid = zeros(self.ls_elem_grid.shape, dtype=int)
        ls_ielem_grid = ma.masked_array(elem_grid, self.ls_mask)
        ls_ielem_grid[self.ls_grid_ix] = enum_elems
        return ls_ielem_grid

    xelems_arr_ix = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_xelems_arr_ix(self):
        '''Get the array indices of the enriched elements within the flattened arrays
        elem_dof_map, elem_X_map, elem_x_map
        '''
        return self.ls_ielem_grid[self.xelems_grid_ix]

    # element indices within a flattened grid
    ls_elems = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_ls_elems(self):
        return self.ls_elem_grid.compressed()

    # element indices within a flattened grid
    elems = Property(depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_elems(self):
        return self.intg_elem_grid.compressed()

    #-------------------------------------------------------------------------
    # Flattened element arrays - dof_map, X_map, x_map
    #-------------------------------------------------------------------------
    elem_dof_map = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_elem_dof_map(self):
        '''Get the element dof map
        should be - intg_elem_dof_map
        '''
        ls_elem_dof_map = self.ls_elem_dof_map

        # exclude the enriched elements. The enumeration
        #
        #
        xelems_mask = zeros(ls_elem_dof_map.shape, dtype=bool)
        xelems_mask[...] = False
        xelems_mask[self.xelems_arr_ix, ...] = True

        ma_elem_dof_map = ma.masked_array(ls_elem_dof_map, xelems_mask)
        elem_dof_map = ma_elem_dof_map.compressed()
        n_dof = ma_elem_dof_map.shape[1]
        return elem_dof_map.reshape(self.n_active_elems, n_dof)

    elem_xdof_map = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_elem_xdof_map(self):
        '''Just the enriched dofs of the current elem_dof_map
        '''
        n_e_dofs = self.fets_eval.parent_fets.n_e_dofs
        return self.elem_dof_map[:, n_e_dofs:]

    # The full enumeration map including the elements enriched by children
    #
    ls_elem_dof_map = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_ls_elem_dof_map(self):
        return self.dof_enum_data[0]

    #---------------------------------------------------------------------
    # Methods delegated to the parent subdomain.
    #---------------------------------------------------------------------
    dof_vertex_X_grid = Property

    def _get_dof_vertex_X_grid(self):
        return self.fe_grid.dof_vertex_X_grid

    ls_parent_elems = Property

    def _get_ls_parent_elems(self):
        '''Get the array indices of elements active in this domain
        elements within the parent domain'''
        if isinstance(self.fe_grid, FEGrid):
            n_grid_elems = self.fe_grid.n_grid_elems
            elem_map = arange(int(n_grid_elems)).reshape(self.fe_grid.shape)
        elif isinstance(self.fe_grid, FELSDomain):
            elem_map = self.fe_grid.ls_ielem_grid
        return elem_map[self.ls_grid_ix]

    dof_enum_data = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_dof_enum_data(self):
        '''
        generates  elem_dof_map and n_dofs
        returns the array (ls_elems, dofs)
        '''
        #
        ls_elem_dof_map = copy(
            self.fe_grid.ls_elem_dof_map[self.ls_parent_elems])

        # number of added dofs
        n_xdofs = self.fets_eval.n_e_dofs - self.fets_eval.parent_fets.n_e_dofs

        # This is ugly hack for rough version
        # @todo
        # the number of enriching dofs must be specified explicitly in
        # the fets_eval - used for enrichment.
        #
        if (n_xdofs / 2) > self.fets_eval.parent_fets.n_e_dofs:
            #(array([[0,1,2,3,4,5,6,7,8,9,10,11]]),10)
            return (array([[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]), 10)

        # x_slice stands for number of enriched dofs in a single node.
        #
        n_elems = ls_elem_dof_map.shape[0]
        n_nodal_dofs = self.fe_grid.fets_eval.n_nodal_dofs
        n_nodes = self.fe_grid.fets_eval.n_dof_r

        elem_xdof_map = copy(ls_elem_dof_map).reshape(
            n_elems, n_nodes, n_nodal_dofs)
        elem_xdof_map = elem_xdof_map[
            :, :, self.fets_eval.x_slice].reshape(n_elems, n_xdofs)

        # special case for selective integration - no enrichment. returns
        # the original dofs and zero n_dofs
        #
        if self.fets_eval.__class__ == self.fe_grid.fets_eval.__class__:
            return (ls_elem_dof_map, 0)  # without additional dofs

        # compress the dof enumeration to be a consecutive sequence
        # of integers without gaps
        #
        xidx = sort(unique(elem_xdof_map.flatten()))
        last_idx = 0
        for i in range(len(xidx)):
            shrink = xidx[i] - last_idx
            if shrink > 0:
                # shrink everything behind shrink
                elem_xdof_map[where(elem_xdof_map > i)] -= shrink
            last_idx = xidx[i] + 1

        elem_xdof_map += self.dof_offset
        #
        # return an array of dof numbers - the original dofs of the parent
        # domain are reused by the xfe_subdomain.
        #
        return (hstack([ls_elem_dof_map, elem_xdof_map]), xidx.shape[0])

    elem_X_map = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_elem_X_map(self):
        ls_elem_X_map = self.ls_elem_X_map
        xelems_mask = zeros(ls_elem_X_map.shape, dtype=bool)
        xelems_mask[...] = False
        xelems_mask[self.xelems_arr_ix, ...] = True
        ma_elem_X_map = ma.masked_array(ls_elem_X_map, xelems_mask)
        n_elems = len(self.elems)
        n_X, n_dim = ma_elem_X_map.shape[1:]
        return ma_elem_X_map.compressed().reshape(n_elems, n_X, n_dim)

    elem_x_map = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_elem_x_map(self):
        ls_elem_x_map = self.ls_elem_x_map
        xelems_mask = zeros(ls_elem_x_map.shape, dtype=bool)
        xelems_mask[...] = False
        xelems_mask[self.xelems_arr_ix, ...] = True
        ma_elem_x_map = ma.masked_array(ls_elem_x_map, xelems_mask)
        n_elems = len(self.elems)
        n_X, n_dim = ma_elem_x_map.shape[1:]
        return ma_elem_x_map.compressed().reshape(n_elems, n_X, n_dim)

    ls_elem_X_map = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_ls_elem_X_map(self):
        return self.fe_grid.ls_elem_X_map[self.ls_parent_elems]

    ls_elem_x_map = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_ls_elem_x_map(self):
        return self.fe_grid.ls_elem_x_map[self.ls_parent_elems]

    #-------------------------------------------------------------------------
    # Element as an object. It is shipped within the spatial context
    # to the element procedures so that additional information can be
    # obtained at the level of the material model. For example
    # nonlocal values.
    # @todo - screen the cases where it is used.
    #
    elements = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_elements(self):
        elem_dof_map = self.elem_dof_map
        elem_X_map = self.elem_X_map
        elem_x_map = self.elem_x_map
        return [MElem(dofs=dofs, point_X_arr=point_X_arr, point_x_arr=point_x_arr)
                for dofs, point_X_arr, point_x_arr
                in zip(elem_dof_map, elem_X_map, elem_x_map)]

    #-------------------------------------------------------------------------
    # Methods required by XDOTS
    #-------------------------------------------------------------------------
    # @todo - rename and simplify
    #
    X_dof_arr = Property

    def _get_X_dof_arr(self):
        return self.fe_grid.X_dof_arr

    elem_node_map = Property

    def _get_elem_node_map(self):
        return self.fe_grid.elem_node_map

    dof_node_ls_values = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_dof_node_ls_values(self):
        '''Return the values of the level set
        '''
        X_arr = self.X_dof_arr
        ls_function = self.ls_function
        n_dims = X_arr.shape[1]

        vect_fn = frompyfunc(ls_function, n_dims, 1)
        ls_vertex_grid = sign(vect_fn(*X_arr.T))

        # select the affected elements from the ls_vertex_grid

        elem_node_map = self.elem_node_map

        return ls_vertex_grid[elem_node_map[self.elems]]

    def get_cell_point_X_arr(self, elem):
        return self.fe_grid.get_cell_point_X_arr(elem)

    ls_intersection_r = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_ls_intersection_r(self):
        '''Get the intersection points in local coordinates
        '''
        # fe_grid_slice.r_i # intersecting points
        i_elements = self.elems
        el_pnts = []
        for elem in i_elements:
            inter_pts = []
            # X_mtx = self.elements[elem].get_X_mtx() # skips deactivated
            # elements
            X_mtx = self.get_cell_point_X_arr(elem)
            dim = X_mtx.shape[1]  # TODO:merge 1 and 2d
            if dim == 1:
                r_coord = get_intersect_pt(self.ls_fn_r, (0., X_mtx))
                if r_coord != None:
                    inter_pts.append([r_coord])
            elif dim == 2:
                for c_coord in [-1., 1.]:
                    args = (c_coord, X_mtx)
                    s_coord = get_intersect_pt(self.ls_fn_s, args)
                    r_coord = get_intersect_pt(self.ls_fn_r, args)
                    if s_coord != None:
                        inter_pts.append([c_coord, s_coord])
                    if r_coord != None:
                        inter_pts.append([r_coord, c_coord])
            elif dim == 3:
                raise NotImplementedError('not available for 3D yet')
            el_pnts.append(inter_pts)
        return array(el_pnts)

    def ls_fn_r(self, r, s, X_mtx):  # TODO:dimensionless treatment
        ls_function = self.ls_function
        X_pnt = self.fe_grid.fets_eval.map_r2X([r, s], X_mtx)
        if X_pnt.shape[0] == 1:
            Y = 0.
        else:
            Y = X_pnt[1]
        return ls_function(X_pnt[0], Y)

    def ls_fn_s(self, s, r, X_mtx):
        ls_function = self.ls_function
        X_pnt = self.fe_grid.fets_eval.map_r2X([r, s], X_mtx)
        return ls_function(X_pnt[0], X_pnt[1])

    def ls_fn_X(self, X, Y):
        return self.ls_function(X, Y)

    shape_changed = Event

    @on_trait_change('fe_grid.shape')
    def set_shape_change(self):
        self.shape_changed = True

    shape = Property

    def _get_shape(self):
        return self.fe_grid.shape

    n_dofs = Property

    def _get_n_dofs(self):
        #n_xdofs = self.fets_eval.n_e_dofs - self.fets_eval.parent_fets.n_e_dofs
        # return n_xdofs *  self.n_active_elems
        return self.dof_enum_data[1]

    idx_active_elems = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_idx_active_elems(self):
        return arange(self.n_active_elems)

    n_active_elems = Property(
        depends_on='shape_changed,+ls_changed,idx_mask_changed')

    @cached_property
    def _get_n_active_elems(self):
        return len(self.elems)

    def apply_on_ip_grid(self, fn, ip_mask):
        '''
        Apply the function fn over the first dimension of the array.
        @param fn: function to apply for each ip from ip_mask and each element. 
        @param ip_mask: specifies the local coordinates within the element.     
        '''
        X_el = self.elem_X_map
        # test call to the function with single output - to get the shape of
        # the result.
        out_single = fn(ip_mask[0], X_el[0])
        out_grid_shape = (X_el.shape[0], ip_mask.shape[0], ) + out_single.shape
        out_grid = zeros(out_grid_shape)

        for el in range(X_el.shape[0]):
            for ip in range(ip_mask.shape[0]):
                out_grid[el, ip, ...] = fn(ip_mask[ip], X_el[el])

        return out_grid
