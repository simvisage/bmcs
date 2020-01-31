

from numpy import \
    zeros, float_, ix_, array, vstack, hstack
from traits.api import \
    Array, provides, \
    Tuple, WeakRef, Delegate, Property, cached_property
from ibvpy.core.i_tstepper_eval import \
    ITStepperEval
from ibvpy.core.rtrace_eval import RTraceEval
from ibvpy.fets.i_fets_eval import IFETSEval
from mathkit.matrix_la.sys_mtx_array import SysMtxArray

from .dots_eval import DOTSEval


#-----------------------------------------------------------------------------
# Integrator for a simple 1D domain.
#-----------------------------------------------------------------------------
@provides(ITStepperEval)
class DOTSUnstructuredEval(DOTSEval):
    '''
    Domain with uniform FE-time-step-eval.
    '''
    ip_scheme = Property(Tuple, depends_on='sdomain.+structure_changed')

    @cached_property
    def _get_ip_scheme(self):
        '''Get the integration points for the whole domain.
        '''
        ip_coords_list = []
        ip_weights_list = []
        n_ip_list = []
        parent_domain = self.sdomain.parent

        sctx = self.sdomain.new_scontext()
        for p, refinement_params, fe_child_domain in self.sdomain.fe_subgrids_params:
            sctx.parent_elem = parent_domain[p]
            sctx.refinement_params = refinement_params
            for e_id, elem in enumerate(fe_child_domain.elements):
                sctx.elem = elem
                sctx.X = elem.get_X_mtx()
                sctx.x = elem.get_x_mtx()
                ip_coords, ip_weights = self.fets_eval.get_ip_scheme(sctx)
                # register the number of integration points, their coordinates and weights
                # in the list
                n_ip_list.append(ip_coords.shape[0])
                ip_coords_list.append(ip_coords)
                ip_weights_list.append(ip_weights)
        return array(n_ip_list, dtype='int_'), vstack(ip_coords_list), hstack(ip_weights_list)

    def get_state_array_size(self):

        # The overall size is just a n_elem times the size of a single element
        #
        return 0

    state_array = Property(
        Array(float), depends_on='sdomain.+structure_changed')

    @cached_property
    def _get_state_array(self):
        n_ip_arr, ip_coords_arr, ip_weights_arr = self.ip_scheme
        n_ip = ip_coords_arr.shape[0]
        mp_arr_size = self.fets_eval.get_mp_state_array_size(None)
        return zeros((n_ip, mp_arr_size), dtype='float_')

    cache_geo_matrices = False

    def setup(self, sctx):

        sctx = self.sdomain.new_scontext()
        self.sctx = sctx
        n_ip_arr, ip_coords_arr, ip_weights_arr = self.ip_scheme

        ndofs = self.sdomain.n_dofs
        self.F_int = zeros(ndofs, float_)

        # Run the setup of sub-evaluator
        #
        ip_offset = 0
        for mp_state_array in self.state_array:
            sctx.mats_state_array = mp_state_array
            self.fets_eval.mats_eval.setup(sctx)

        # Setup the system matrix
        self.k_arr = zeros((self.sdomain.shape,
                            self.sdomain.elem_dof_map.shape[1],
                            self.sdomain.elem_dof_map.shape[1]), dtype='float_')

        self.B_mtx_grid = None
        self.J_det_grid = None
        if self.cache_geo_matrices:
            # Calculate the B matrices for all the elements.
            self.B_mtx_grid = self.sdomain.apply_on_ip_grid(self.fets_eval.get_B_mtx,
                                                            self.fets_eval.ip_coords)
            self.J_det_grid = self.sdomain.apply_on_ip_grid(self.fets_eval.get_J_det,
                                                            self.fets_eval.ip_coords)

    def get_corr_pred(self, sctx, u, du, tn, tn1):

        n_ip_arr, ip_coords_arr, ip_weights_arr = self.ip_scheme

        self.F_int[:] = 0.0
        self.k_arr[...] = 0.0

        B_mtx_grid = None
        J_det_grid = None

        ip_offset = 0
        k_list = []
        for e_id, (elem, n_ip) in enumerate(zip(self.sdomain.elements, n_ip_arr)):
            ip_coords = ip_coords_arr[ip_offset: ip_offset + n_ip]
            ip_weights = ip_weights_arr[ip_offset: ip_offset + n_ip]
            ix = elem.get_dof_map()
            sctx.elem = elem
            sctx.elem_state_array = self.state_array[ip_offset: ip_offset + n_ip].flatten(
            )
            sctx.X = elem.get_X_mtx()
            if self.cache_geo_matrices:
                B_mtx_grid = self.B_mtx_grid[e_id, ...]
                J_det_grid = self.J_det_grid[e_id, ...]
            f, k = self.fets_eval.get_corr_pred(sctx, u[ix_(ix)], du[ix_(ix)],
                                                tn, tn1,
                                                B_mtx_grid=B_mtx_grid,
                                                J_det_grid=J_det_grid,
                                                ip_coords=ip_coords,
                                                ip_weights=ip_weights)

            self.k_arr[e_id] = k
            self.F_int[ix_(ix)] += f
            ip_offset += n_ip

        return self.F_int, SysMtxArray(mtx_arr=self.k_arr, dof_map_arr=self.sdomain.elem_dof_map)


class RTraceEvalUDomainUnstructuredFieldVar(RTraceEval):
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
