
#from sys_matrix import SysSparseMtx, SysDenseMtx
import unittest

from numpy import array, zeros, arange, array_equal, hstack, dot
from numpy import array, zeros, arange, array_equal, sqrt
from scipy.linalg import solve, norm

from .coo_mtx import COOSparseMtx
from .dense_mtx import DenseMtx
from .sys_mtx_assembly import SysMtxAssembly


# bar clamped at the left end and loaded at the right end
def get_bar_mtx_array(shape, k=10):
    '''Get an array of matrices and dof_maps corresponding to
    a bar discretization
    '''
    el_mtx = array([[k, -k],
                    [-k, k]], dtype='float_')

    el_mtx_arr = array([el_mtx for i in range(shape)], dtype=float)
    el_dof_map = array([arange(shape),
                        arange(shape) + 1], dtype=int).transpose()

    return el_dof_map, el_mtx_arr


class TestSysMtxSolver(unittest.TestCase):
    '''
    Test the matrix functionality required for Kling the
    system matrix, construction of the sparsity map and solution 
    of the system.

    The sparse matrices are constructed using the matrix Kly 
    list.
    '''

    def setUp(self):
        '''
        Set up a simple 3x3 matrix corresponding to a tension bar
        connected in the middle node.

        The first DOF is factored prescribed to be zero - 
        the off-diagonal terms are therefore deleted 
        (only 1 remains at the diagonal)
        '''
        n_dofs = 3
        shape = 2
        el_mtx = array([[1, -1],
                        [-1, 1]], dtype='float_')

        el_mtx_arr = array([el_mtx for i in range(shape)], dtype=float)
        el_dof_map = array([arange(n_dofs - 1),
                            arange(n_dofs - 1) + 1], dtype=int).transpose()

        # Kly object
        self.sys_K = SysMtxAssembly()

        self.sys_K.add_mtx_array(dof_map_arr=el_dof_map,
                                 mtx_arr=el_mtx_arr)
        self.sys_K.register_constraint(0)
        self.rhs = zeros(n_dofs)
        self.rhs[-1] = 1.

        # solve the system using standard scipy methods
        # on a full matrix
        #
        la_mtx = array([[1, 0, 0],
                        [0, 2, -1],
                        [0, -1, 1]], dtype=float)
        la_rhs = array([0, 0, 1], dtype=float)

        self.la_u = solve(la_mtx, la_rhs)

    def test_coo_sparse_mtx(self):
        '''Construct the coordinate sparsity map and matrix and solve it.
        '''
        u = self.sys_K.solve(self.rhs, matrix_type='coord')
        self.assertTrue(array_equal(u, self.la_u))

    def test_dense_mtx(self):
        '''Construct the dense matrix and solve it.
        '''
        u = self.sys_K.solve(self.rhs, matrix_type='dense')
        self.assertTrue(array_equal(u, self.la_u))


class TestSysMtxConstraints(unittest.TestCase):
    '''
    Test functionality connected with the application of
    constraints.
    '''

    def test_bar1(self):
        '''Clamped bar loaded at the right end with unit displacement
        [00]-[01]-[02]-[03]-[04]-[05]-[06]-[07]-[08]-[09]-[10]
        'u[0] = 0, u[10] = 1'''
        K = SysMtxAssembly()
        dof_map, mtx_arr = get_bar_mtx_array(shape=10)
        K.add_mtx_array(dof_map_arr=dof_map, mtx_arr=mtx_arr)
        K.register_constraint(a=0,  u_a=0.)  # clamped end
        K.register_constraint(a=10, u_a=1.)
        K.register_constraint(a=10, u_a=1.)
        # add load
        R = zeros(K.n_dofs)
        # system solver
        u = K.solve(R)
        # expected solution
        u_ex = array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                     dtype=float)
        difference = sqrt(norm(u - u_ex))
        self.assertAlmostEqual(difference, 0)

    def test_bar2(self):
        '''Clamped bar composed of two linked bars loaded at the right end
        [00]-[01]-[02]-[03]-[04]-[05]-[06]-[07]-[08]-[09]-[10]
        [11]-[12]-[13]-[14]-[15]-[16]-[17]-[18]-[19]-[20]-[21]
        u[0] = 0, u[5] = u[16], R[-1] = R[21] = 10
        '''
        K = SysMtxAssembly()
        dof_map1, mtx_arr1 = get_bar_mtx_array(shape=10)
        K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
        dof_map2, mtx_arr2 = get_bar_mtx_array(shape=10)
        # Note the dof map of the second br must be adjusted to start
        # the DOF enumeration with 3
        n_dofs1 = K.n_dofs
        K.add_mtx_array(dof_map_arr=dof_map2 + n_dofs1, mtx_arr=mtx_arr2)
        # add constraints
        K.register_constraint(a=0, u_a=0.)  # clamped end
        K.register_constraint(a=5, alpha=[1], ix_a=[16])
        # add load
        R = zeros(K.n_dofs)
        R[-1] = 10
        # system solver
        u = K.solve(R)
        # expected solution
        u_ex = array([0., 1., 2., 3., 4., 5., 5., 5., 5., 5., 5.,
                      5., 5., 5., 5., 5., 5., 6., 7., 8., 9., 10.],
                     dtype=float)
        difference = sqrt(norm(u[1] - u_ex[1]))
        self.assertAlmostEqual(difference, 0)

        #
        # '---------------------------------------------------------------'
        # 'Clamped bar composed of two linked bars control displ at right'
        # 'u[0] = 0, u[5] = u[16], u[21] = 1'
        # Remove the load and put a unit displacement at the right end
        # Note, the load is irrelevant in this case and will be rewritten
        #
        K.register_constraint(a=21, u_a=1)
        # system solver
        u = K.solve(R, matrix_type='dense')
        # expected solution
        u_ex = array([0.,   1 / 10.,  2 / 10., 3 / 10., 4 / 10., 5 / 10., 5 / 10., 5 / 10.,
                      5 / 10., 5 / 10.,  5 / 10.,  5 /
                      10.,  5 / 10.,  5 / 10.,  5 / 10.,
                      5 / 10.,  5 / 10.,  6 / 10.,  7 / 10.,  8 / 10.,  9 / 10.,  1.],
                     dtype=float)
        difference = sqrt(norm(u[-1] - u_ex[-1]))
        self.assertAlmostEqual(difference, 0)
        #

    def test_bar3(self):
        '''Clamped bar with recursive constraints (load at right end)
        [0]-[1]-[2]-[3]
        u[1] = 0.2 * u[2], u[2] = 0.2 * u[3], R[3] = 10
        '''
        K = SysMtxAssembly()
        dof_map, mtx_arr = get_bar_mtx_array(shape=3)
        K.add_mtx_array(dof_map_arr=dof_map, mtx_arr=mtx_arr)
        K.register_constraint(a=0, u_a=0.)  # clamped end
        K.register_constraint(a=1, alpha=[0.5], ix_a=[2])
        K.register_constraint(a=2, alpha=[1], ix_a=[3])
        # add load
        R = zeros(K.n_dofs)
        R[3] = 1
        # system solver
        # K.apply_constraints(R)
        u = K.solve(R)
        # expected solution
        u_ex = array([-0.,  0.1, 0.2, 0.2],
                     dtype=float)
        difference = sqrt(norm(u - u_ex))
        self.assertAlmostEqual(difference, 0)
        #
        # '---------------------------------------------------------------'
        # 'Clamped bar with recursive constraints (displ at right end)'
        # 'u[1] = 0.5 * u[2], u[2] = 1.0 * u[3], u[3] = 1'
        K.register_constraint(a=3, u_a=1)
        # system solver
        u = K.solve(R)
        # expected solution
        u_ex = array([0.,  0.5, 1,  1], dtype=float)
        difference = sqrt(norm(u - u_ex))
        self.assertAlmostEqual(difference, 0)
        #

    def test_bar4(self):
        '''Clamped bar 3 domains, each with 2 elems (displ at right end)
        [0]-[1]-[2] [3]-[4]-[5] [6]-[7]-[8]
        u[0] = 0, u[2] = u[3], u[5] = u[6], u[8] = 1'''
        K = SysMtxAssembly()
        dof_map1, mtx_arr1 = get_bar_mtx_array(shape=2)
        K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
        dof_map2, mtx_arr2 = get_bar_mtx_array(shape=2)
        K.add_mtx_array(dof_map_arr=dof_map2 + 3, mtx_arr=mtx_arr2)
        dof_map3, mtx_arr3 = get_bar_mtx_array(shape=2)
        K.add_mtx_array(dof_map_arr=dof_map3 + 6, mtx_arr=mtx_arr3)
        # add constraints
        K.register_constraint(a=0, u_a=0.)  # clamped end
        K.register_constraint(a=2, alpha=[1], ix_a=[3])
        K.register_constraint(a=5, alpha=[1], ix_a=[6])
        K.register_constraint(a=8, u_a=1.)  # loaded end
        # add load
        R = zeros(K.n_dofs)
        # system solver
        u = K.solve(R)
        # expected solution
        u_ex = array([0., 1 / 6.,  1 / 3., 1 / 3., 1 / 2., 2 / 3., 2 / 3.,  5 / 6.,  1.],
                     dtype=float)
        difference = sqrt(norm(u - u_ex))
        self.assertAlmostEqual(difference, 0)

    def test_bar5(self):
        '''Clamped bar with 4 elements. Elements 2-4 are reinforced
        with another bar with 3 elements
        [0]-[1]-[2]-[3]-[4]
            [5]-[6]-[7]'''
        # 'u[0] = 0, u[1] = u[5], u[3] = u[7], u[4] = 1'
        K = SysMtxAssembly()
        dof_map1, mtx_arr1 = get_bar_mtx_array(shape=4)
        K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
        dof_map2, mtx_arr2 = get_bar_mtx_array(shape=2)
        K.add_mtx_array(dof_map_arr=dof_map2 + 5, mtx_arr=mtx_arr2)
        # add constraints
        K.register_constraint(a=0, u_a=0.)  # clamped end
        K.register_constraint(a=1, alpha=[1], ix_a=[5])
        K.register_constraint(a=3, alpha=[1], ix_a=[7])
        K.register_constraint(a=4, u_a=1.)  # loaded end
        # add load
        R = zeros(K.n_dofs)
        # system solver
        u = K.solve(R)
        # expected solution
        u_ex = array([0., 1 / 3.,  0.5,
                      2 / 3.,  1.,
                      1 / 3.,  0.5,         2 / 3.],
                     dtype=float)
        difference = sqrt(norm(u - u_ex))
        self.assertAlmostEqual(difference, 0)
        #

    def test_bar6(self):
        '''Clamped bar with 4 elements. Elements 2-4 are reinforced 
        with another bar with 1 element linked proportianally
        [0]-[1]-[2]-[3]-[4]
              [5]-[6]
        u[0] = 0, u[1] = u[5], u[3] = u[7], u[4] = 1'''
        K = SysMtxAssembly()
        dof_map1, mtx_arr1 = get_bar_mtx_array(shape=4)
        K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
        dof_map2, mtx_arr2 = get_bar_mtx_array(shape=1)
        K.add_mtx_array(dof_map_arr=dof_map2 + 5, mtx_arr=mtx_arr2)
        # add constraints
        K.register_constraint(a=0, u_a=0.)  # clamped end
        K.register_constraint(a=5, alpha=[0.5, 0.5], ix_a=[1, 2])
        K.register_constraint(a=6, alpha=[0.5, 0.5], ix_a=[2, 3])
        K.register_constraint(a=4, u_a=1.)  # loaded end
        # add load
        R = zeros(K.n_dofs)
        # system solver
        u = K.solve(R)
        # expected solution
        u_ex = array([-0.,  0.3, 0.5, 0.7, 1.,  0.4,  0.6], dtype=float)
        difference = sqrt(norm(u - u_ex))
        self.assertAlmostEqual(difference, 0)

    def test_bar7(self):
        '''Two clamped beams link in parallel
        and loaded by force at right end
        [5]-[6]-[7]-[8]-[9]
        [0]-[1]-[2]-[3]-[4]
        u[5] = u[0], u[0] = 0, u[4] = u[9], R[4] = 1'''
        K = SysMtxAssembly()
        dof_map1, mtx_arr1 = get_bar_mtx_array(shape=4)
        K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
        dof_map2, mtx_arr2 = get_bar_mtx_array(shape=4)
        K.add_mtx_array(dof_map_arr=dof_map2 + 5, mtx_arr=mtx_arr2)
        # add constraints
        K.register_constraint(a=5, alpha=[1], ix_a=[0])
        K.register_constraint(a=0)  # clamped end
        K.register_constraint(a=4, alpha=[0.5], ix_a=[9])
        # add load
        R = zeros(K.n_dofs)
        # load at the coupled end nodes is doubled
        R[9] = 1
        R[4] = 1
        # system solver
#        print 'K\n',K
#        print 'R\n',R
        u = K.solve(R)
#        print 'K\n',K
#        print 'R\n',R
#        print 'u\n',u
        # expected solution
        u_ex = array([-0., 0.06, 0.12,  0.18,  0.24,  0.,
                      0.12,  0.24, 0.36,  0.48],
                     dtype=float)
        difference = sqrt(norm(u - u_ex))
        self.assertAlmostEqual(difference, 0)

    def test_bar8(self):
        '''Overruling of a non-zero constraint by a zero-constraint
        applied for corner and edge constrains.
        [0]-[1]
        u[0] = 0, u[1] = 0, u[1] = 4'''
        K = SysMtxAssembly()
        dof_map1, mtx_arr1 = get_bar_mtx_array(shape=1)
        K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
        # add constraints
        K.register_constraint(a=0, u_a=0)  # clamped end
        K.register_constraint(a=1, u_a=4)  # should be ignored
        K.register_constraint(a=1, u_a=0)  # should remain
        # add load
        R = zeros(K.n_dofs)
        # load at the coupled end nodes is doubled
        # system solver
        u = K.solve(R)
        # expected solution
        u_ex = array([-0., -0.], dtype=float)
        self.assertAlmostEqual(u[1], u_ex[1])

    def test_bar9(self):
        '''Zero terms  at the diagonal. combined with zero-value constraints
        (simulating the deactivation of elements)
        [0]-[1]
        u[0] = 0, u[1] = 0'''
        K = SysMtxAssembly()

        dof_map1, mtx_arr1 = get_bar_mtx_array(shape=1, k=0.)
        K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
        # add constraints
        K.register_constraint(a=0, u_a=0)  # clamped end
        K.register_constraint(a=1, u_a=0)  # clamped end
        # add load
        R = zeros(K.n_dofs)
        R[1] = 10.  # irrelevant
        # load at the coupled end nodes is doubled
        # system solver
        u = K.solve(R)
        # expected solution
        u_ex = array([-0., -0.], dtype=float)
        self.assertAlmostEqual(u[1], u_ex[1])

    def test_bar10(self):
        '''Zero terms  at the diagonal. combined with zero-value constraints
        (simulating the deactivation of elements)
        [0]-[1]-[2]-[3]-[4]  [5]-[6]
        u[0] = 0, u[4] = u[5]  u[6] = 0, K[5,6] = 0, R[3] = 1'''
        K = SysMtxAssembly()
        dof_map1, mtx_arr1 = get_bar_mtx_array(shape=4)
        K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
        dof_map2, mtx_arr2 = get_bar_mtx_array(shape=1, k=0.)
        K.add_mtx_array(dof_map_arr=dof_map2 + 5, mtx_arr=mtx_arr2)
        # add constraints
        K.register_constraint(a=6, u_a=0.)
        K.register_constraint(a=0)  # clamped end
        K.register_constraint(a=4, alpha=[1], ix_a=[5])
        # add load
        R = zeros(K.n_dofs)
        # load at the coupled end nodes is doubled
        R[3] = 1
        # system solver
        u = K.solve(R)
        # expected solution
        u_ex = array([-0., 0.1, 0.2, 0.3, 0.3, 0.3, 0.],
                     dtype=float)
        for uu, ue in zip(u, u_ex):
            self.assertAlmostEqual(uu, ue)

    def test_bar11(self):
        '''XXXXXX Zero terms  at the diagonal. combined with zero-value constraints
        (simulating the deactivation of elements)
        [0]-[1]-[2]-[3]-[4]  [5]-[6]
        u[0] = 0, u[4] = u[5]  u[6] = 0, K[5,6] = 0, R[3] = 1'''
        K = SysMtxAssembly()
        dof_map2, mtx_arr2 = get_bar_mtx_array(shape=1, k=0)
        K.add_mtx_array(dof_map_arr=dof_map2, mtx_arr=mtx_arr2)
        dof_map1, mtx_arr1 = get_bar_mtx_array(shape=4)
        K.add_mtx_array(dof_map_arr=dof_map1 + 2, mtx_arr=mtx_arr1)
        # add constraints
        K.register_constraint(a=6, u_a=0.)
        K.register_constraint(a=2, alpha=[1], ix_a=[1])
        K.register_constraint(a=0)  # clamped end
        # add load
        R = zeros(K.n_dofs)
        # load at the coupled end nodes is doubled
        R[3] = - 1
        # system solver
        u = K.solve(R)
        # expected solution
        u_ex = array([-0.,  -0.3, -0.3, -0.3, -0.2, -0.1, -0.],
                     dtype=float)
        for uu, ue in zip(u, u_ex):
            self.assertAlmostEqual(uu, ue)
