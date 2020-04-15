'''
Created on Mar 25, 2020

@author: rch
'''

import numpy as np
import traits.api as tr


class ILinAlgSys(tr.Interface):
    pass


class LinAlgSolve(tr.HasStrictTraits):
    '''Example of a linear equation system

    Solve the system of equations of the form

    :math
        A u = b_0 - b_t

    To define the boundary conditions prescribing the 
    values on the left-hand-side a sequential
    inclusion of the condition is managed by this class.
    Thus, the registration of the condition is performed
    first to register the variables affected by the 
    problem.
    '''
    o_I = tr.Array(np.int_)

    n_dofs = tr.Property(depends_on='dof_map')

    @tr.cached_property
    def _get_n_dofs(self):
        return np.max(self.o_I) + 1

    A = tr.Array(np.float_)

    R_t = tr.Property(tr.Array(np.float), depends_on='dof_map')

    @tr.cached_property
    def _get_R_t(self):
        return np.zeros((self.n_dofs,), np.float_)

    R_0 = tr.Property(tr.Array(np.float), depends_on='dof_map')

    @tr.cached_property
    def _get_R_0(self):
        return np.zeros((self.n_dofs,), np.float_)

    # The dependency graph

    constraints = tr.List
    '''Constraints are stored as a list of arrays.
    '''

    def register_constraint(self, a_U, U_a=0, tf=lambda t: t,
                            b_alpha=[], alpha_b=[]):
        '''Register the prescribed values of u as linear 
        transformations - mappings
        '''
        self.constraints.append((a_U, U_a, tf, b_alpha, alpha_b))

    t_n1 = tr.Float(1.0)

    U_arr = tr.Property(depends_on='t_n1')

    @tr.cached_property
    def _get_U_arr(self):
        '''Get the value of the constraint for the current step.
        '''
        carr = np.array(self.constraints)
        U_arr = carr[:, 1]
        tfarr = carr[:, 2]
        val_arr = np.array(
            [tf(u)
             for tf, u in zip(tfarr, U_arr)], dtype=np.float_)
        return val_arr

    def apply_constraints(self):
        carr = np.array(self.constraints)
        a_arr = np.array(carr[:, 0], dtype=np.int_)
        A_aa = self.A[a_arr, a_arr]
        self.A[a_arr, :] = 0
        self.A[:, a_arr] = 0
        self.A[a_arr, a_arr] = A_aa
        self.R_0[:] = np.einsum('i,ij->j', self.U_arr, self.A[a_arr, :])

    def solve(self):
        '''Solve the system
        '''
        return np.linalg.solve(self.A, self.R_0 - self.R_t)


if __name__ == '__main__':
    las = LinAlgSolve(
        A=[[1, 2, 3], [2, 4, 1], [3, 1, 5]],
        o_I=[0, 1, 2],
    )
    B = np.copy(las.A)
    B[(1, 2), (1, 2)]

    # Todo
    # Right hand side
    # Bundled calls to a sparse format matrix
    # Sparse matrix
    # How to merge the protocol with the compressed row
    # Define the time functions generally as piecewise linear

    las.register_constraint(0, 1)
    las.register_constraint(1, 1)
    las.apply_constraints()
    print(las.constraints)
    print(las.A)
    print(las.solve())
