
from math import fabs

from numpy import allclose, arange, eye, linalg, ones, ix_, array, zeros, \
    hstack, meshgrid, vstack, dot, newaxis, c_, r_, copy, where, \
    ones, append, unique, compress, array_equal, allclose
from traits.api import \
    HasTraits, Int, Array, Property, cached_property, List, Trait, Dict, \
    Any, Bool, Float

from .coo_mtx import COOSparseMtx
from .dense_mtx import DenseMtx
from .sys_mtx_array import SysMtxArray


class Constraint(HasTraits):

    u_a = Property(Float)

    def _set_u_a(self, value):
        self._u_a = value * self._freeze_coeff

    def _get_u_a(self):
        return self._u_a

    # Freeze the constraint by setting the coefficient to zero
    # applied for overruled constraints
    #
    _freeze_coeff = 1.0

    def freeze(self):
        self._freeze_coeff = 0.0

    a = Int(-1)
    alpha = Array(float)
    ix_a = Array(int)

    def __str__(self):
        return 'a = %d, u_a = %g, alpha = %s, ix_a = %s' % \
            (self.a, self.u_a, str(self.alpha), str(self.ix_a))


class SysMtxAssembly(HasTraits):
    '''Class intermediating between the time stepper and the linear algebra solver.

    Handling of constraints.

    Constraints are added by the method

        register_constraint( self, a, u_a, alpha, ix_a )

    They are registered in the trait list called

        constraints = List

    The inclusion of the constraints is performed on the mapped matrices stored
    in the sys_mtx_arrays. It is possible that contributions to the position in
    the system matrix are stored in several sys_mtx_arrays. Therefore, the
    constraint must be included in all matrix arrays currently included.

    Such an operation requires a search throuth the matrices, which is realized
    using the

        numpy.where

    method. In order to avoid repeated invokation of where the acccess to the
    affected positions in the sys_mtx_array can be cached. The state management
    recognizes the attributes

        value_changed = Bool
        structure_changed = Bool

    Thus, an adaptive strategy or spatial domain mut indicate the change in
    the structure explicitly by setting

        sys_assembly = True

    The sys_assembly maintains a property attribute

        cached_constraints = Property()

    containing the result of the where operations delivering the addresses
    of the format

        [el, row]
    '''
    # list of matrix arrays
    #
    sys_mtx_arrays = List

    # list of matrix arrays
    #
    link_matrices = List

    # list of applied constraints
    #
    constraints = List

    # Dictionary for fast access
    #
    _c = Dict

    # matrix type to be used when solving the system
    #
    matrix_type = Trait('coord',
                        {'dense': DenseMtx,
                         'coord': COOSparseMtx})

    # number of degrees of freedom
    #
    n_dofs = Property(Int)

    def _get_n_dofs(self):
        n_dofs = 0
        # must iterate over all arrays to get the maximum number of DOFs
        # the enumeration is done within the spatial domain object.
        #
        for mtx_array in self.sys_mtx_arrays:
            n_dofs = max(n_dofs, mtx_array.n_dofs)
        return n_dofs

    def add_mtx_array(self, mtx_arr, dof_map_arr):
        '''Add an array of matrices with the dof map
        '''
        sys_mtx_arr = SysMtxArray(dof_map_arr=dof_map_arr,
                                  mtx_arr=mtx_arr)
        self.sys_mtx_arrays.append(sys_mtx_arr)

    def add_mtx(self, mtx, dof_map=None):
        '''Add a single matrix with the dof map
        '''
        if dof_map == None:
            dof_map = arange(mtx.shape[0])

        sys_mtx_array = SysMtxArray(dof_map_arr=dof_map[None, ...],
                                    mtx_arr=mtx[None, ...]
                                    )
        self.sys_mtx_arrays.append(sys_mtx_array)
        return sys_mtx_array

    def add_link_mtx(self, mtx, dof_map=None):
        '''Add a single matrix with the dof map
        '''
        if dof_map == None:
            dof_map = arange(mtx.shape[0])

        link_mtx = SysMtxArray(dof_map_arr=dof_map[None, ...],
                               mtx_arr=mtx[None, ...]
                               )
        self.link_matrices.append(link_mtx)

    debug = Bool(False)

    def register_constraint(self, a, u_a=0, alpha=[], ix_a=[]):

        constraint = self._c.get(a, None)

        if constraint == None:

            # constraint does not exist yet, add it
            #
            constraint = Constraint(a=a, u_a=u_a, alpha=alpha, ix_a=ix_a)
            self.constraints.append(constraint)
            self._c[a] = constraint

            if self.debug:
                print('new constraint: a:', a, 'u_a', u_a, 'alpha', alpha, 'ix_a', ix_a)

        else:
            # duplicate specification, if it is identical with the
            # previous one simply ignore it, otherwise throw and exception.
            #
            # Handle the "corner" and "edge" constraints. The rule is, the
            # zero-constraint overrules the non-zero or a link constraint.
            if constraint.u_a == 0 and constraint.alpha.shape[0] == 0 and \
                    constraint.ix_a.shape[0] == 0:
                # ignore the non-zero constraint
                # the zero-constraint prevails
                #
                constraint.freeze()
                if self.debug:
                    print('frozen constraint:', constraint)
            elif u_a == 0 and len(alpha) == 0 and len(ix_a) == 0:
                # set the existing constraint to zero-constraint
                # ignore the non-zero value and the coefficients
                constraint.freeze()
                constraint.u_a = u_a
                constraint.alpha = alpha
                constraint.ix_a = ix_a
                if self.debug:
                    print('frozen constraint:', constraint)
            elif not constraint.u_a == u_a or \
                    not allclose(constraint.alpha, alpha, rtol=1e-4) or \
                    not array_equal(constraint.ix_a, ix_a):
                raise ValueError('contradicting constraint definition:\n' \
                    'a = %d, u = %f, alpha = %s, ix_a = %s\n' \
                    'previous constraint:\n%s' % (
                        a, u_a, alpha, ix_a, constraint))

        return constraint

    sorted_constraints = Property(depends_on='constraints[]')

    @cached_property
    def _get_sorted_constraints(self):

        sorted_c = self.constraints
        for i in range(1000):
            sorted_c, swapped_values = self._get_simply_sorted_c(sorted_c)
            if swapped_values == False:
                return sorted_c
        # exception - the cyclic specification of constraints
        raise ValueError('maximum number of reorderings (1000) reached\n' \
            'this is probably due to cyclic constraints specification')

    def _get_simply_sorted_c(self, constraints):
        # first test if one of the indices in ix_a is already in _c
        sorted_a = []
        sorted_c = []
        swapped_values = False
        for constraint in constraints:
            # suggest the last position in the traversed elements
            pos = len(sorted_a)
            for ix in constraint.ix_a:
                try:
                    found_pos = sorted_a.index(ix)
                    if found_pos < pos:
                        # The found position is below the last one
                        # swap required. Insert the current constraint
                        # before the found one
                        pos = found_pos
                        swapped_values = True
                except ValueError:
                    pass
            sorted_a.insert(pos, constraint.a)
            sorted_c.insert(pos, constraint)

        return sorted_c, swapped_values

    cached_addresses = Property(depends_on='constraints[]')

    @cached_property
    def _get_cached_addresses(self):
        cached_addresses = []
        for constraint in self.sorted_constraints:
            ix_maps = self._get_ix_maps(constraint)
            cached_addresses.append(ix_maps)
        return cached_addresses

    def solve(self, rhs=None, matrix_type=None, check_pos_def=False):
        '''Solve the system of equations using a specified
        type of matrix format
        '''

        if self.debug:
            print('SysMtxAssembly:', id(self))

        if rhs is None and self._rhs is None:
            raise ValueError('No right hand side available')

        if not rhs is None:
            self.apply_constraints(rhs)

        if matrix_type:
            self.matrix_type = matrix_type

        mtx = self.matrix_type_(assemb=self)
        return mtx.solve(self._rhs, check_pos_def)

    def reset(self):
        self.sys_mtx_arrays = []
        self.constraints = []
        self.link_matrices = []
        self._c = {}
        self.rhs = None

    def reset_mtx(self):
        self.sys_mtx_arrays = []
        self.rhs = None

    _rhs = Any

    def print_constraints(self):
        # apply the constraints
        for constraint in self.constraints:
            print(constraint)

    def apply_constraints(self, rhs):
        # apply the constraints
        for constraint, ix_maps in zip(self.sorted_constraints,
                                       self.cached_addresses):
            if self.debug:
                print('applying constraint', constraint)

            self._apply_constraint(rhs, constraint, ix_maps)

        self._rhs = rhs  # rhs that should be used for solving the system

    def get_sys_mtx_arrays(self):
        '''Return the complete list of matrix arrays including constraints.
        '''
        return self.sys_mtx_arrays + self.link_matrices

    def __str__(self):
        '''Create dense matrix and print it
        '''
        return str(DenseMtx(assemb=self))

    mtx = Property

    def _get_mtx(self):
        return DenseMtx(assemb=self).mtx

    def _get_ix_maps(self, constraint):
        '''
        Add the constraint associated with the dof number `a'
        '''
        c = constraint

        dof_ix_array = self.get_dof_ix_array(c.a)

        a, u_a, alpha, ix_a = (c.a, c.u_a, c.alpha, c.ix_a)

        # link

        if type(alpha) == list:
            alpha = array(alpha, dtype=float)
        if type(ix_a) == list:
            ix_a = array(ix_a, dtype=int)

        if alpha.shape[0] == 0:

            # no links - manipulates only a single equation
            #
            return (alpha, None, dof_ix_array, None, None, None, None, None, None, None)

        # find out which non-zero entries has the affected dofs
        ix_K, K_n_a = self._get_col_subvector(dof_ix_array)

        ix_orig_layout = hstack([ix_K, ix_a])

#            K_n_a2 = zeros( ix_layout.shape[0], dtype = float )
#            K_n_a2[:ix_K.shape[0]] = K_n_a
#            alpha2 = zeros( ix_layout.shape[0], dtype = float )
#            alpha2[ix_K.shape[0]:] = alpha

        ix_mask = ix_orig_layout != a  # deactivate the constrained dof

        prev_same_i_list = []
        alpha_same_i_list = []
        for i, ix in enumerate(ix_K):
            prev_same_i = where(ix_K == ix)[0][0]
            prev_same_i_list.append(prev_same_i)
            if prev_same_i < i:
                #                    K_n_a2[prev_same_i] += K_n_a2[i]
                ix_mask[i] = False  # deactivate the added value

            if ix in ix_a:
                alpha_same_i = where(ix_a == ix)[0][0]
                alpha_same_i_list.append(alpha_same_i)
                alpha_same_i += ix_K.shape[0]
#                    alpha2[i] = alpha2[ alpha_same_i ]
                ix_mask[alpha_same_i] = False
            else:
                alpha_same_i_list.append(0)

#            K_n_a     = compress( ix_mask, K_n_a2 )
        ix_layout = compress(ix_mask, ix_orig_layout)
#            alpha     = compress( ix_mask, alpha2 )

        # Get the size of the link matrix
        link_mtx_sz = ix_layout.shape[0] + 1

        #
        # Add the a-th column to the right hand side
        # Note, K_n_a is not the full column, it contains only the
        # rows of the dof involved in the linear combination (indices)

#            P_a = rhs[a]
#            rhs[a] = 0

#            self._add_col_to_vector( dof_ix_array, rhs, factor = - u_a )

#            K_a_a = self._get_diag_elem( dof_ix_array )

        # After remembering the submatrices in K_aa, and K_na
        # the constrained row and column can be zeroed
        #
#            self._zero_rc( dof_ix_array )

        # Redistribute the load applied to the constrained dof
        #
#            rhs[ ix_layout ] += alpha.transpose() * P_a

        # fill the dof map of the additional link matrix
        #
        link_dof_map = zeros((link_mtx_sz), dtype=int)
        link_dof_map[0] = a
        link_dof_map[1:] = ix_layout

        # fill the link matrix itself
        #
        # @todo: split -
        link_mtx = zeros((link_mtx_sz, link_mtx_sz), dtype=float)
#        link_mtx[0,0] = - K_a_a
#        link_mtx[0,1:] = K_a_a * alpha.transpose()
#        link_mtx[1:,0] = link_mtx[0,1:].transpose()
#        K_n_a_alpha = dot( K_n_a[:,None], alpha[None,:] )
#        link_mtx[1:,1:] = K_n_a_alpha + K_n_a_alpha.transpose()

        # Add the link matrix
        #
        self.add_link_mtx(mtx=link_mtx, dof_map=link_dof_map)

        return (alpha, ix_a, dof_ix_array, link_dof_map,
                ix_orig_layout, ix_layout, ix_mask, link_mtx,
                array(prev_same_i_list, dtype='int_'), array(alpha_same_i_list, dtype='int_'))

    def _apply_constraint(self, rhs, constraint, ix_map):
        '''
        Add the constraint associated with the dof number `a'
        '''
        c = constraint

        alpha, ix_a, dof_ix_array, link_dof_map, ix_orig_layout, ix_layout, ix_mask, link_mtx, prev_same_i_array, alpha_same_i_array = ix_map

        a, u_a, alpha, ix_a = (c.a, c.u_a, c.alpha, c.ix_a)

        # link

        if alpha.shape[0] == 0:

            # constraint affect only a single equation
            # simplified handling.
            #
            # Add the a-th column to the right hand side
            # Note, K_n_a is not the full column, it contains only the
            # rows of the dof involved in the linear combination (indices)

            P_a = rhs[a]
            rhs[a] = 0

            self._add_col_to_vector(dof_ix_array, rhs, factor=-u_a)

            K_a_a = self._get_diag_elem(dof_ix_array)

            if fabs(K_a_a) < (1.0e-5):
                K_a_a = 1.
                rhs[a] = -u_a

            # After remembering the submatrices in K_aa, and K_na
            # the constrained row and column can be zeroed
            #
            self._zero_rc(dof_ix_array)

            self._add_diag_elem(dof_ix_array, -K_a_a)

        else:

            # find out which non-zero entries has the affected dofs
            ix_K, K_n_a = self._get_col_subvector(dof_ix_array)

            K_n_a2 = zeros(ix_orig_layout.shape[0], dtype=float)
            K_n_a2[:ix_K.shape[0]] = K_n_a

            alpha2 = zeros(ix_orig_layout.shape[0], dtype=float)
            alpha2[ix_K.shape[0]:] = alpha

            # ix_mask = ix_layout != a # deactivate the constrained dof

            for i, ix in enumerate(ix_K):
                #prev_same_i = where( ix_K == ix )[0][0]
                prev_same_i = prev_same_i_array[i]
                if prev_same_i < i:
                    K_n_a2[prev_same_i] += K_n_a2[i]

                if ix in ix_a:
                    #alpha_same_i = where( ix_a == ix )[0][0]
                    alpha_same_i = alpha_same_i_array[i]
                    alpha_same_i += ix_K.shape[0]
                    alpha2[i] = alpha2[alpha_same_i]

            K_n_a = compress(ix_mask, K_n_a2)
            alpha = compress(ix_mask, alpha2)

            # Get the size of the link matrix
            link_mtx_sz = alpha.shape[0] + 1

            #
            # Add the a-th column to the right hand side
            # Note, K_n_a is not the full column, it contains only the
            # rows of the dof involved in the linear combination (indices)

            P_a = rhs[a]
            rhs[a] = 0

            self._add_col_to_vector(dof_ix_array, rhs, factor=-u_a)

            K_a_a = self._get_diag_elem(dof_ix_array)

            # This has not been thoroughly tested
            # If a DOF is in the air, it can be linked to an existing
            # dof via kinematic constraint. This is used upon deactivation
            # of elements.
            #
            if fabs(K_a_a) < (1.0e-5):
                K_a_a = 1.

            # After remembering the submatrices in K_aa, and K_na
            # the constrained row and column can be zeroed
            #
            self._zero_rc(dof_ix_array)

            # Redistribute the load applied to the constrained dof
            #
            rhs[ix_layout] += alpha.transpose() * P_a

            # fill the link matrix itself
            #
            #link_mtx = zeros( (link_mtx_sz, link_mtx_sz), dtype = float )
            link_mtx[0, 0] = -K_a_a
            link_mtx[0, 1:] = K_a_a * alpha.transpose()
            link_mtx[1:, 0] = link_mtx[0, 1:].transpose()
            K_n_a_alpha = dot(K_n_a[:, None], alpha[None, :])
            link_mtx[1:, 1:] = K_n_a_alpha + K_n_a_alpha.transpose()

            # Add the link matrix
            #
            # This has been moved to the cached part _get_ix_maps
            # - the link matrices are only modified in-place.
            #
            #self.add_mtx( mtx = link_mtx, dof_map = link_dof_map )

        return

    def get_dof_ix_array(self, dof):
        '''Get the access to the posisions containing values
        related to the dof
        '''
        return [mtx_array.get_dof_ix_array(dof)
                for mtx_array in self.get_sys_mtx_arrays()]

    def _zero_rc(self, dof_ix_arrays):
        for mtx_array, dof_ix_array in zip(self.get_sys_mtx_arrays(), dof_ix_arrays):
            mtx_array._zero_rc(dof_ix_array)

    def _add_col_to_vector(self, dof_ix_arrays, F, factor):
        for mtx_array, dof_ix_array in zip(self.get_sys_mtx_arrays(), dof_ix_arrays):
            mtx_array._add_col_to_vector(dof_ix_array, F, factor)

    def _get_col_subvector(self, dof_ix_arrays):
        idx_arr = array([], dtype=int)
        val_arr = array([], dtype=float)
        for mtx_array, dof_ix_array in zip(self.get_sys_mtx_arrays(), dof_ix_arrays):
            idx_seg, val_seg = mtx_array._get_col_subvector(dof_ix_array)
            idx_arr = append(idx_arr, idx_seg)
            val_arr = append(val_arr, val_seg)
        return idx_arr, val_arr

    def _get_diag_elem(self, dof_ix_arrays):
        K_dof_dof = 0
        for mtx_array, dof_ix_array in zip(self.get_sys_mtx_arrays(), dof_ix_arrays):
            K_dof_dof += mtx_array._get_diag_elem(dof_ix_array)
        return K_dof_dof

    def _add_diag_elem(self, dof_ix_arrays, K_aa):
        '''Set value of a diagonal element. Find the first occurrence
        of the element value and put it there
        '''
        for mtx_array, dof_ix_array in zip(self.get_sys_mtx_arrays(), dof_ix_arrays):
            el, r_ix = dof_ix_array
            if el.shape[0] > 0:
                mtx_array._add_diag_elem(dof_ix_array, K_aa)
                break
