
from numpy import \
    zeros, float_, ix_,  repeat, arange, array, dot
from traits.api import \
    Array, Bool, provides, \
    Instance, Int, \
    WeakRef, Delegate, Property, cached_property, Dict
from traitsui.api import \
    Item, View
from traitsui.menu import \
    OKButton, CancelButton

from ibvpy.core.i_tstepper_eval import \
    ITStepperEval
from ibvpy.core.rtrace_eval import RTraceEval
from ibvpy.core.tstepper_eval import \
    TStepperEval
from ibvpy.fets.i_fets_eval import IFETSEval
from ibvpy.mesh.i_fe_uniform_domain import IFEUniformDomain
from mathkit.matrix_la.sys_mtx_array import SysMtxArray


@provides(ITStepperEval)
class DOTSEval(TStepperEval):
    '''
    Domain with uniform FE-time-step-eval.
    Integrator for a general regular domain.
    '''
    sdomain = Instance(IFEUniformDomain)

    fets_eval = Property(Instance(IFETSEval), depends_on='sdomain.fets_eval')

    @cached_property
    def _get_fets_eval(self):
        return self.sdomain.fets_eval

    def new_cntl_var(self):
        return zeros(self.sdomain.n_dofs, float_)

    def new_resp_var(self):
        return zeros(self.sdomain.n_dofs, float_)

    def new_tangent_operator(self):
        '''
        Return the tangent operator used for the time stepping
        '''
        return SysMtxArray()

    cache_geo_matrices = Bool(True)

    # cached zeroed array for element stiffnesses
    k_arr = Property(Array, depends_on='sdomain.changed_structure')

    @cached_property
    def _get_k_arr(self):
        n_e, n_e_dofs = self.sdomain.elem_dof_map_unmasked.shape
        return zeros((n_e, n_e_dofs, n_e_dofs), dtype='float_')

    F_int = Property(Array, depends_on='sdomain.changed_structure')

    @cached_property
    def _get_F_int(self):
        return zeros(self.sdomain.n_dofs, float_)

    B_mtx_grid = Property(
        Array, depends_on='sdomain.changed_structure,sdomain.+changed_geometry')

    @cached_property
    def _get_B_mtx_grid(self):
        return self.sdomain.apply_on_ip_grid_unmasked(self.fets_eval.get_B_mtx,
                                                      self.fets_eval.ip_coords)

    J_det_grid = Property(
        Array, depends_on='sdomain.changed_structure,sdomain.+changed_geometry')

    @cached_property
    def _get_J_det_grid(self):
        return self.sdomain.apply_on_ip_grid_unmasked(self.fets_eval.get_J_det,
                                                      self.fets_eval.ip_coords)

    state_array_size = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_state_array_size(self):

        # The overall size is just a n_elem times the size of a single element
        #
        n_elems = self.sdomain.n_elems
        self.e_arr_size = self.fets_eval.get_state_array_size()
        return n_elems * self.e_arr_size

    ip_offset = Property(depends_on='sdomain.changed_structure')

    def _get_ip_offset(self):
        n_elems = self.sdomain.n_elems
        return arange(n_elems + 1, dtype=int) * self.fets_eval.n_gp

    # temporary alias - method deprecated - use property instead
    def get_state_array_size(self):
        return 0

    state_array = Property(
        Array, depends_on='sdomain.changed_structure,sdomain.+changed_geometry')

    @cached_property
    def _get_state_array(self):
        state_array = zeros((self.state_array_size,), dtype='float_')
        e_arr_size = self.fets_eval.get_state_array_size()

        sctx = self.sdomain.domain.new_scontext()
        # Run the setup of sub-evaluator
        #
        for e_id, elem in zip(self.sdomain.idx_active_elems, self.sdomain.elements):
            sctx.elem = elem
            sctx.elem_state_array = state_array[
                e_id * e_arr_size: (e_id + 1) * e_arr_size]
            self.fets_eval.setup(sctx)
        return state_array

    def get_corr_pred(self, u, du, tn, tn1, F_int, *args, **kw):

        # in order to avoid allocation of the array in every time step
        # of the computation
        k_arr = self.k_arr
        k_arr[...] = 0.0
        e_arr_size = self.fets_eval.get_state_array_size()

        # build in control that there is at least one active elem
        if self.cache_geo_matrices and len(self.sdomain.elements) != 0:
            B_mtx_grid = self.B_mtx_grid
            J_det_grid = self.J_det_grid

        Be_mtx_grid = None
        Je_det_grid = None

        state_array = self.state_array

        tstepper = self.sdomain.tstepper
        U = tstepper.U_k
        d_U = tstepper.d_U

        # generic arguments to be pushed through the loop levels
        args_fets = []
        kw_fets = {}
        U_avg_k = kw.get('eps_avg', None)
        if U_avg_k != None:
            u_avg_arr = U_avg_k[self.sdomain.elem_dof_map_unmasked]

        for e_id, elem in zip(self.sdomain.idx_active_elems, self.sdomain.elements):

            ix = elem.get_dof_map()

            sctx.elem = elem
            sctx.elem_state_array = state_array[
                e_id * e_arr_size: (e_id + 1) * e_arr_size]
            sctx.X = elem.get_X_mtx()
            sctx.x = elem.get_x_mtx()
            if self.cache_geo_matrices:
                Be_mtx_grid = B_mtx_grid[e_id, ...]
                Je_det_grid = J_det_grid[e_id, ...]

            if U_avg_k != None:
                kw_fets['eps_avg'] = u_avg_arr[e_id, ...]

            f, k = self.fets_eval.get_corr_pred(sctx, U[ix_(ix)], d_U[ix_(ix)],
                                                tn, tn1,
                                                B_mtx_grid=Be_mtx_grid,
                                                J_det_grid=Je_det_grid,
                                                *args_fets, **kw_fets)

            k_arr[e_id] = k
            F_int[ix_(ix)] += f

        return SysMtxArray(mtx_arr=k_arr, dof_map_arr=self.sdomain.elem_dof_map_unmasked)

    def map_u(self, sctx, U):
        ix = sctx.elem.get_dof_map()
        u = U[ix]
        return u

    # @todo: Jakub remove this - specific to two field problems
    # Should be a tracer assocated with the element.
    def get_eps_m(self, sctx, u):
        X_mtx = sctx.X
        r_pnt = sctx.loc
        B_mtx = self.fets_eval.get_B_mtx(r_pnt, X_mtx)
        eps = dot(B_mtx, u)
        return array([[eps[0], eps[2]], [eps[2], eps[1]]])

    # Specific to two field problems
    # Should be a tracer associated with the element.
    def get_eps_f(self, sctx, u):
        X_mtx = sctx.X
        r_pnt = sctx.loc
        B_mtx = self.fets_eval.get_B_mtx(r_pnt, X_mtx)
        eps = dot(B_mtx, u)
        return array([[eps[3], eps[5]], [eps[5], eps[4]]])

    # @todo: Jakub remove this - specific to two field problems
    # Should be a tracer assocated with the element.
    def get_u_m(self, sctx, u):
        N_mtx = self.fets_eval.get_N_mtx(sctx.loc)
        return dot(N_mtx, u)[:2]

    # @todo: Jakub remove this - specific to two field problems
    # Should be a tracer assocated with the element.
    def get_u_f(self, sctx, u):
        N_mtx = self.fets_eval.get_N_mtx(sctx.loc)
        return dot(N_mtx, u)[2:]

    rte_dict = Property(Dict, depends_on='fets_eval')

    @cached_property
    def _get_rte_dict(self):
        rte_dict = {}

        # @todo: Jakub remove this - specific to two field problems
        # Should be a tracer assocated with the element.
        rte_dict.update({'u_m': RTraceEvalUDomainFieldVar(eval=self.get_u_m, ts=self, u_mapping=self.map_u),
                         'u_f': RTraceEvalUDomainFieldVar(eval=self.get_u_f, ts=self, u_mapping=self.map_u),
                         #                          'u_rm' : RTraceEvalUDomainFieldVar( eval = self.get_u, ts = self, u_mapping = self.map_u ),
                         #                          'u_rf' : RTraceEvalUDomainFieldVar( eval = self.get_u, ts = self, u_mapping = self.map_u ),
                         'eps_m': RTraceEvalUDomainFieldVar(eval=self.get_eps_m, ts=self, u_mapping=self.map_u),
                         'eps_f': RTraceEvalUDomainFieldVar(eval=self.get_eps_f, ts=self, u_mapping=self.map_u),
                         })
        for key, eval in list(self.fets_eval.rte_dict.items()):

            rte_dict[key] = RTraceEvalUDomainFieldVar(name=key,
                                                      u_mapping=self.map_u,
                                                      eval=eval,
                                                      fets_eval=self.fets_eval)
        return rte_dict

    def get_vtk_r_arr(self, idx):
        return self.fets_eval.vtk_r_arr

    def get_vtk_pnt_ip_map(self, e_id):
        return self.fets_eval.vtk_pnt_ip_map

    def get_vtk_X(self, position):
        '''Get the discretization points based on the fets_eval
        associated with the current domain.
        '''
        if position == 'int_pnts':
            ip_arr = self.fets_eval.ip_coords

        pts = []
        dim_slice = self.fets_eval.dim_slice
        for e in self.sdomain.elements:
            X = e.get_X_mtx()
            if dim_slice:
                X = X[:, dim_slice]
                if position == 'int_pnts':
                    ip_arr = ip_arr[:, dim_slice]
            if position == 'nodes':
                pts += list(self.fets_eval.get_vtk_r_glb_arr(X))
            elif position == 'int_pnts':
                pts += list(self.fets_eval.get_vtk_r_glb_arr(X, ip_arr))
        pts_array = array(pts, dtype='float_')
        return pts_array

    debug_cell_data = Bool(False)

    # @todo - comment this procedure`
    def get_vtk_cell_data(self, position, point_offset, cell_offset):
        if position == 'nodes':
            subcell_offsets, subcell_lengths, subcells, subcell_types = \
                self.fets_eval.vtk_node_cell_data
        elif position == 'int_pnts':
            subcell_offsets, subcell_lengths, subcells, subcell_types = \
                self.fets_eval.vtk_ip_cell_data

        if self.debug_cell_data:
            print('subcell_offsets')
            print(subcell_offsets)
            print('subcell_lengths')
            print(subcell_lengths)
            print('subcells')
            print(subcells)
            print('subcell_types')
            print(subcell_types)

        n_subcells = subcell_types.shape[0]
        n_cell_points = self.n_cell_points
        subcell_size = subcells.shape[0] + n_subcells

        if self.debug_cell_data:
            print('n_cell_points', n_cell_points)
            print('n_cells', self.n_cells)

        vtk_cell_array = zeros((self.n_cells, subcell_size), dtype=int)

        idx_cell_pnts = repeat(True, subcell_size)

        if self.debug_cell_data:
            print('idx_cell_pnts')
            print(idx_cell_pnts)

        idx_cell_pnts[subcell_offsets] = False

        if self.debug_cell_data:
            print('idx_cell_pnts')
            print(idx_cell_pnts)

        idx_lengths = idx_cell_pnts == False

        if self.debug_cell_data:
            print('idx_lengths')
            print(idx_lengths)

        point_offsets = arange(self.n_cells) * n_cell_points
        point_offsets += point_offset

        if self.debug_cell_data:
            print('point_offsets')
            print(point_offsets)

        vtk_cell_array[:, idx_cell_pnts] = point_offsets[
            :, None] + subcells[None, :]
        vtk_cell_array[:, idx_lengths] = subcell_lengths[None, :]

        if self.debug_cell_data:
            print('vtk_cell_array')
            print(vtk_cell_array)

        #active_cells = self.sdomain.idx_active_elems
        n_active_cells = self.sdomain.n_active_elems

        if self.debug_cell_data:
            print('n active cells')
            print(n_active_cells)

        cell_offsets = arange(n_active_cells, dtype=int) * subcell_size
        cell_offsets += cell_offset
        vtk_cell_offsets = cell_offsets[:, None] + subcell_offsets[None, :]

        if self.debug_cell_data:
            print('vtk_cell_offsets')
            print(vtk_cell_offsets)

        vtk_cell_types = zeros(self.n_cells * n_subcells, dtype=int).reshape(self.n_cells,
                                                                             n_subcells)
        vtk_cell_types += subcell_types[None, :]

        if self.debug_cell_data:
            print('vtk_cell_types')
            print(vtk_cell_types)

        return vtk_cell_array.flatten(), vtk_cell_offsets.flatten(), vtk_cell_types.flatten()

    n_cells = Property(Int)

    def _get_n_cells(self):
        '''Return the total number of cells'''
        return self.sdomain.n_active_elems

    n_cell_points = Property(Int)

    def _get_n_cell_points(self):
        '''Return the number of points defining one cell'''
        return self.fets_eval.n_vtk_r

    traits_view = View(Item('fets_eval', style='custom', show_label=False),
                       resizable=True,
                       height=0.8,
                       width=0.8,
                       buttons=[OKButton, CancelButton],
                       kind='subpanel',
                       scrollable=True,
                       )


class RTraceEvalUDomainFieldVar(RTraceEval):
    fets_eval = WeakRef(IFETSEval)

    # @TODO Return the parametric coordinates of the element covering the element domain
    #
    vtk_r_arr = Delegate('fets_eval')
    field_entity_type = Delegate('fets_eval')
    dim_slice = Delegate('fets_eval')
    get_vtk_r_glb_arr = Delegate('fets_eval')
    n_vtk_r = Delegate('fets_eval')
    field_faces = Delegate('fets_eval')
    field_lines = Delegate('fets_eval')
    get_state_array_size = Delegate('fets_eval')
    n_vtk_cells = Delegate('fets_eval')
    vtk_cell_data = Delegate('fets_eval')
    vtk_ip_cell_data = Delegate('fets_eval')
