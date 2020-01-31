
from numpy import allclose, arange, eye, linalg, ones, ix_, array, zeros, \
    hstack, meshgrid, vstack, dot, newaxis, c_, r_, copy, where, \
    ones
from traits.api import HasTraits, Array, Property, cached_property, Instance, \
    Delegate, Any
from traitsui.api \
    import View, Item, TabularEditor
from traitsui.tabular_adapter \
    import TabularAdapter


#from sys_mtx_assembly import SysMtxAssembly
class ArrayAdapter(TabularAdapter):

    columns = Property

    def _get_columns(self):
        n_columns = getattr(self.object, self.name).shape[1]
        cols = [(str(i), i) for i in range(n_columns)]
        return [('i', 'index')] + cols

    font = 'Courier 10'
    alignment = 'right'
    format = '%6.2f'
    index_text = Property

    def _get_index_text(self):
        return str(self.row)

tabular_editor = TabularEditor(
    adapter=ArrayAdapter())


class DenseMtx(HasTraits):

    '''Dense matrix with the interface of a sparse matrix assembly.

    Used for debugging and performance comparison of sparse solvers.
    '''
    assemb = Any

    mtx = Property

    def _get_mtx(self):
        n_dofs = self.assemb.n_dofs
        sys_mtx = zeros([n_dofs, n_dofs], dtype=float)
        # loop over the list of matrix arrays
        for sys_mtx_arr in self.assemb.get_sys_mtx_arrays():
            # loop over the matrix array
            for dof_map, mtx in zip(sys_mtx_arr.dof_map_arr,
                                    sys_mtx_arr.mtx_arr):
                sys_mtx[ix_(dof_map, dof_map)] += mtx
        return sys_mtx

    def solve(self, rhs):
        #tf_solve_s = time()
        u_vct = linalg.solve(self.mtx, rhs)
        #tf_solve_e = time()
        #dif_solve = tf_solve_e - tf_solve_s
        # print "Full Solve: %8.2f sec" %dif_solve
        return u_vct

    def __str__(self):
        '''String representation - delegate to matrix'''
        return str(self.mtx)

    view = View(Item('mtx', editor=tabular_editor, show_label=False),
                resizable=True,
                scrollable=True,
                buttons=['OK', 'Cancel'],
                width=1.0, height=0.5)
