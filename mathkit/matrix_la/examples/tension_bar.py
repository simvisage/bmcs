
#from sys_matrix import SysSparseMtx, SysDenseMtx
from numpy import array, zeros, arange, array_equal, hstack, dot

from mathkit.matrix_la.coo_mtx import COOSparseMtx
from mathkit.matrix_la.dense_mtx import DenseMtx
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly


# bar clamped at the left end and loaded at the right end
def get_bar_mtx_array(shape):
    '''Get an array of matrices and dof_maps corresponding to
    a bar discretization
    '''
    el_mtx = array([[10, -10],
                    [-10, 10]], dtype='float_')

    el_mtx_arr = array([el_mtx for i in range(shape)], dtype=float)
    el_dof_map = array([arange(shape),
                        arange(shape) + 1], dtype=int).transpose()

    return el_dof_map, el_mtx_arr


def bar1():
    print('---------------------------------------------------------------')
    print('Clamped bar loaded at the right end with unit displacement')
    print('[00]-[01]-[02]-[03]-[04]-[05]-[06]-[07]-[08]-[09]-[10]')
    print('u[0] = 0, u[10] = 1')
    K = SysMtxAssembly()
    dof_map, mtx_arr = get_bar_mtx_array(shape=10)
    K.add_mtx_array(dof_map_arr=dof_map, mtx_arr=mtx_arr)
    K.register_constraint(a=0,  u_a=0.)  # clamped end
    K.register_constraint(a=10, u_a=1.)
    K.register_constraint(a=10, u_a=1.)
    K_dense = DenseMtx(assemb=K)
    R = zeros(K.n_dofs)
    print('K\n', K_dense)
    print('R\n', R)
    print('K_arrays')
    for i, sys_mtx_array in enumerate(K.sys_mtx_arrays):
        print('i\n', sys_mtx_array.mtx_arr)
    K.apply_constraints(R)
    K_dense = DenseMtx(assemb=K)
    print('K\n', K_dense)
    print('R\n', R)
    print('K_arrays')
    for i, sys_mtx_array in enumerate(K.sys_mtx_arrays):
        print('i\n', sys_mtx_array.mtx_arr)
    print('u =',  K.solve(R))
    print()


def bar2():
    print('---------------------------------------------------------------')
    print('Clamped bar composed of two linked bars loaded at the right end')
    print('[00]-[01]-[02]-[03]-[04]-[05]-[06]-[07]-[08]-[09]-[10]')
    print('[11]-[12]-[13]-[14]-[15]-[16]-[17]-[18]-[19]-[20]-[21]')
    print('u[0] = 0, u[5] = u[16], R[-1] = R[21] = 10')
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
    print('u =', K.solve(R))
    print()
    print('---------------------------------------------------------------')
    print('Clamped bar composed of two linked bars control displ at right')
    print('u[0] = 0, u[5] = u[16], u[21] = 1')
    # Remove the load and put a unit displacement at the right end
    # Note, the load is irrelevant in this case and will be rewritten
    #
    K.register_constraint(a=21, u_a=1)
    print('u =', K.solve(R))
    print()


def bar3():
    print('---------------------------------------------------------------')
    print('Clamped bar with recursive constraints (load at right end)')
    print('[0]-[1]-[2]-[3]')
    print('u[1] = 0.5 * u[2], u[2] = 1 * u[3], R[3] = 11')
    K = SysMtxAssembly()
    dof_map, mtx_arr = get_bar_mtx_array(shape=3)
    K.add_mtx_array(dof_map_arr=dof_map, mtx_arr=mtx_arr)
    K.register_constraint(a=0, u_a=0.)  # clamped end
    K.register_constraint(a=1, alpha=[0.5], ix_a=[2])
    K.register_constraint(a=2, alpha=[1.0], ix_a=[3])
    R = zeros(K.n_dofs)
    R[3] = 1
    K.apply_constraints(R)
    print('u =', K.solve())
    print()
    print('---------------------------------------------------------------')
    print('Clamped bar with recursive constraints (displ at right end)')
    print('u[1] = 0.5 * u[2], u[2] = 1.0 * u[3], u[3] = 1')
    K.register_constraint(a=3, u_a=1)
    print('u =', K.solve(R))
    print()


def bar4():
    print('---------------------------------------------------------------')
    print('Clamped bar 3 domains, each with 2 elems (displ at right end)')
    print('[0]-[1]-[2] [3]-[4]-[5] [6]-[7]-[8]')
    print('u[0] = 0, u[2] = u[3], u[5] = u[6], u[8] = 1')
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
    print('u =', K.solve(R))
    print()
    ####


def bar5():
    print('---------------------------------------------------------------')
    print('Clamped bar with 4 elements. Elements 2-4 are reinforced ')
    print('with another bar with 3 elements')
    print('[0]-[1]-[2]-[3]-[4]')
    print('    [5]-[6]-[7]')
    print('u[0] = 0, u[1] = u[5], u[3] = u[7], u[4] = 1')
    # assemble the matrix
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
    print('u =', K.solve(R))
    print()


def bar6():
    print('---------------------------------------------------------------')
    print('Clamped bar with 4 elements. Elements 2-4 are reinforced ')
    print('with another bar with 1 element linked proportianally')
    print('[0]-[1]-[2]-[3]-[4]')
    print('      [5]-[6]')
    print('u[0] = 0, u[1] = u[5], u[3] = u[7], u[4] = 1')
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
    print('u =', K.solve(R))
    print()
    ####


def bar7():
    print('---------------------------------------------------------------')
    print('Two clamped beams link in parallel')
    print('and loaded by force at right end')
    print('[5]-[6]-[7]-[8]-[9]')
    print('[0]-[1]-[2]-[3]-[4]')
    print('u[5] = u[0], u[0] = 0, u[4] = 0.5 * u[9], R[4] = 1')
    K = SysMtxAssembly()
    dof_map1, mtx_arr1 = get_bar_mtx_array(shape=4)
    K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
    dof_map2, mtx_arr2 = get_bar_mtx_array(shape=4)
    K.add_mtx_array(dof_map_arr=dof_map2 + 5, mtx_arr=mtx_arr2)
    # add load
    R = zeros(K.n_dofs)
    # load at the coupled end nodes is doubled
    R[9] = 1
    R[4] = 1
    # add constraints
    K.register_constraint(a=5, alpha=[1], ix_a=[0])
    K.register_constraint(a=0, u_a=0.)  # clamped end
    K.register_constraint(a=4, alpha=[0.5], ix_a=[9])
    # add load
    R = zeros(K.n_dofs)
    # load at the coupled end nodes is doubled
    R[9] = 1
    R[4] = 1
    print('u =', K.solve(R))
    print()
    ####


def bar8():
    print('---------------------------------------------------------------')
    print('Single clamped element with two constraints')
    print('within a single element')
    print('[0]-[1]')
    print('u[0] = u[1], u[1] = 1')
    K = SysMtxAssembly()
    dof_map1, mtx_arr1 = get_bar_mtx_array(shape=1)
    K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
    # add constraints
    K.register_constraint(a=0, alpha=[1], ix_a=[1])
    K.register_constraint(a=1, u_a=1.)  # clamped end
    # add load
    R = zeros(K.n_dofs)
    print('u =', K.solve(R))
    print()
    ####


def bar9():
    print('---------------------------------------------------------------')
    print('Single clamped element with two constraints')
    print('within a single element')
    print('[0]-[1]-[2]-[3]')
    print('u[0] = u[1], u[1] = 1')
    K = SysMtxAssembly()
    dof_map1, mtx_arr1 = get_bar_mtx_array(shape=3)
    K.add_mtx_array(dof_map_arr=dof_map1, mtx_arr=mtx_arr1)
    # add constraints
    K.register_constraint(a=0)
    K.register_constraint(a=3, u_a=1.)  # clamped end
    # add load
    R = zeros(K.n_dofs)
    print('u =', K.solve(R))
    print()
    K.register_constraint(a=1, alpha=[1], ix_a=[2])
    print('u =', K.solve(R))
    print()
    ####

if __name__ == '__main__':
    bar1()
#    bar2()
#    bar3()
#    bar4()
#    bar5()
#    bar6()
#    bar7()
#    bar8()
#    bar9()
