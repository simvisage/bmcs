
from numpy import \
    array, zeros, array_equal, arange, \
    hstack, sort, where, unique, copy, frompyfunc, sign, fabs as nfabs, \
    logical_and, argwhere, fabs
from scipy.optimize import \
    brentq
from traits.api import \
    Instance, Property, cached_property, \
    provides, Float, \
    Callable, Str, Enum

from ibvpy.dots.xdots_eval import XDOTSEval
from ibvpy.fets.fets_eval import FETSEval
from ibvpy.mesh.i_fe_grid_slice import IFEGridSlice
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
class XFESubDomain(FESubDomain):
    '''Subgrid derived from another grid domain.
    '''

    # Distinguish which part of the level set to take
    #
    domain_tag = Enum('both', 'pos', 'neg')
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
    bls_function = Callable

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
    _slice = Instance(IFEGridSlice)

    # slice property
    slice = Property

    def _set_slice(self, new_slice):
        '''Set the slice by keeping track of the elements included in the subdomain so far. 
        '''
        # remember the element numbers that were in the slice so far.

        old_state_array = None

        if self._slice:
            old_elems = self.elems
            # get the new elemes
            new_elems = new_slice.elems
            # remember the state associated with the elements
            print('old_elems', old_elems)
            print('new_elems', new_elems)
            old_size = old_elems.size
            new_size = new_elems.size

            # if the old array is contained at the beginning of the new array
            # reuse the state array
            if old_size < new_size and array_equal(old_elems, new_elems[:old_size]):
                old_state_array = copy(self.dots.state_array)

        self._slice = new_slice

        # reuse the old state array
        if old_state_array:
            self.dots.state_array[:old_size] = old_state_array

    def _get_slice(self):
        '''Get current slice.
        '''
        return self._slice

    fe_grid = Property

    def _get_fe_grid(self):
        return self._slice.fe_grid

    #---------------------------------------------------------------------
    # Element groups accessed as properties
    #---------------------------------------------------------------------

    # Elements cut by the boundary LS function and by the discontinuity LS
    #
    boundary_elems = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_boundary_elems(self):
        #
        # boundary elements are defined as cells that are
        # contain both positive and negative values of the level set function
        # and are crossed with the domain boundary.
        #
        b_elems_grid = logical_and(nfabs(self.bls_elem_grid) < 4,
                                   nfabs(self.ls_elem_grid) < 4)

        return self._get_elems_from_elem_grid(b_elems_grid)

    interior_elems = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_interior_elems(self):

        # evaluate the boundary operator - with a result boolean array
        # in the corner nodes.
        #
        b_elem_grid = self.bls_elem_grid

        ls_elem_grid = self.ls_elem_grid

        # make the intersection
        #
        elem_grid = logical_and(nfabs(ls_elem_grid) < 4, b_elem_grid == 4)

        return self._get_elems_from_elem_grid(elem_grid)

    def _get_elems_from_elem_grid(self, elem_grid):
        '''Get an array of element offsets based on the from an array
        with boolean identifiers of elements. All elements corresponding
        to the boolean identifiers are returned. Subsequently, the
        array based properties of the domain, like elem_dof_map,
        elem_geo_x and elem_X_geo can be accessed.
        '''
        # get the mask of elements belonging to the level set domain
        #
        elem_idx = argwhere(elem_grid)

        n_dims = elem_idx.shape[1]

        # prepare the index as a sequence of arrays in x, y, and z dimensions
        #
        elem_idx_tuple = tuple([elem_idx[:, i] for i in range(n_dims)])

        # get the enumeration of cells in a n-dim array
        #
        c_g = self.fe_grid.dof_grid.cell_grid.cell_idx_grid

        # select the elements cut by the level set
        #
        return c_g[elem_idx_tuple]

    elems = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_elems(self):
        return hstack([self.interior_elems, self.boundary_elems])

    bls_elem_grid = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_bls_elem_grid(self):

        X_grid = self.fe_grid.dof_grid.cell_grid.vertex_X_grid

        # evaluate the boundary operator - with a result boolean array
        # in the corner nodes.
        #
        b_vertex_grid = sign(self.bls_function(*X_grid))

        return edge_sum(b_vertex_grid)

    ls_elem_grid = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_ls_elem_grid(self):

        X_grid = self.fe_grid.dof_grid.cell_grid.vertex_X_grid

        # find the indices of the elems

        ls_function = self.slice.ls_function

        n_dims = X_grid.shape[0]

        vect_fn = frompyfunc(ls_function, n_dims, 1)
        ls_vertex_grid = sign(vect_fn(*X_grid))

        return array(edge_sum(ls_vertex_grid), dtype=float)

    def deactivate_sliced_elems(self):
        for e in self.elems:
            self.fe_grid.deactivate(e)

    elements = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_elements(self):
        elem_dof_map = self.elem_dof_map
        elem_X_map = self.elem_X_map
        elem_x_map = self.elem_x_map
        return [MElem(dofs=dofs, point_X_arr=point_X_arr, point_x_arr=point_x_arr)
                for dofs, point_X_arr, point_x_arr
                in zip(elem_dof_map, elem_X_map, elem_x_map)]

    elem_dof_map = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_elem_dof_map(self):
        return self.dof_enum_data[0]

    dof_enum_data = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_dof_enum_data(self):
        '''
        generates  elem_dof_map and n_dofs
        TODO - optimize the the algorithm, replace loop with sucessive slices
        to find jump in the enumeration
        TODO - generalize the algorithm from dofs to nodes, this will allow
        to have more extended dofs than parent ones  
        '''
        # (elems, dofs)

        # @todo
        # It would be better to provide fe_grid with
        # several element access methods - specify an array of elements
        # and you get a consistent array of dofs, geo_x and geo_X values.
        #
        dofs = self.fe_grid.dof_grid.cell_dof_map[self.elems]

        sh = dofs.shape

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
        elem_dof_map = copy(dofs).reshape(
            sh[0], self.fets_eval.parent_fets.n_e_dofs)

        elem_xdof_map = (copy(dofs)[:, :, self.fets_eval.x_slice]).reshape(
            sh[0], n_xdofs)

        # special case for selective integration - no enrichment. returns
        # the original dofs and zero n_dofs
        #
        if elem_xdof_map.shape[1] == 0:
            return (dofs, 0)  # without additional dofs

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
        return (hstack([elem_dof_map, elem_xdof_map]), xidx.shape[0])

    elem_X_map = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_elem_X_map(self):
        geo_X = self.fe_grid.geo_grid.elem_X_map[self.elems]
        return geo_X

    elem_x_map = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_elem_x_map(self):
        geo_x = self.fe_grid.geo_grid.elem_x_map[self.elems]
        return geo_x

    dof_node_ls_values = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_dof_node_ls_values(self):

        X_grid = self.fe_grid.dof_grid.cell_grid.point_X_arr
        ls_function = self.slice.ls_function
        n_dims = X_grid.shape[1]

        vect_fn = frompyfunc(ls_function, n_dims, 1)
        ls_vertex_grid = sign(vect_fn(*X_grid.T))

        # select the affected elements from the ls_vertex_grid

        elem_node_map = self.fe_grid.dof_grid.cell_grid.cell_node_map

        return ls_vertex_grid[elem_node_map[self.elems]]

    ls_intersection_r = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_ls_intersection_r(self):
        '''Get the intersection points in local coordinates
        '''
        # fe_grid_slice.r_i # intersecting points
        fe_grid = self.fe_grid
        i_elements = self.elems
        el_pnts = []
        for elem in i_elements:
            inter_pts = []
            # X_mtx = self.elements[elem].get_X_mtx() # skips deactivated
            # elements
            X_mtx = fe_grid.geo_grid.get_cell_point_X_arr(elem)
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
        ls_function = self.slice.ls_function
        X_pnt = self.fe_grid.fets_eval.map_r2X([r, s], X_mtx)
        if X_pnt.shape[0] == 1:
            Y = 0.
        else:
            Y = X_pnt[1]
        return ls_function(X_pnt[0], Y)

    def ls_fn_s(self, s, r, X_mtx):
        ls_function = self.slice.ls_function
        X_pnt = self.fe_grid.fets_eval.map_r2X([r, s], X_mtx)
        return ls_function(X_pnt[0], X_pnt[1])

    n_elems = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_n_elems(self):
        return self.elems.shape[0]

    n_dofs = Property

    @cached_property
    def _get_n_dofs(self):
        #n_xdofs = self.fets_eval.n_e_dofs - self.fets_eval.parent_fets.n_e_dofs
        # return n_xdofs *  self.n_active_elems
        return self.dof_enum_data[1]

    idx_active_elems = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_idx_active_elems(self):
        return arange(self.shape)

    n_active_elems = Property(depends_on='_slice, boundary')

    @cached_property
    def _get_n_active_elems(self):
        return self.shape  # TODO:

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
