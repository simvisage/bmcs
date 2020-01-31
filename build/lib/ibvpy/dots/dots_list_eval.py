
from numpy import \
    zeros, float_
from traits.api import \
    provides, \
    List, \
    WeakRef, Property, cached_property, Dict
from traitsui.api import \
    View
from ibvpy.core.i_tstepper_eval import \
    ITStepperEval
from ibvpy.core.tstepper_eval import \
    TStepperEval
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly


#-----------------------------------------------------------------------------
# Integrator for a simple 1D domain.
#-----------------------------------------------------------------------------
@provides(ITStepperEval)
class DOTSListEval(TStepperEval):

    '''
    Domain with uniform FE-time-step-eval.
    '''

    sdomain = WeakRef('ibvpy.mesh.fe_domain.FEDomain')

    dots_list = List

    def new_cntl_var(self):
        return zeros(self.sdomain.n_dofs, float_)

    def new_resp_var(self):
        return zeros(self.sdomain.n_dofs, float_)

    K = Property

    @cached_property
    def _get_K(self):
        return SysMtxAssembly()

    F_int = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_F_int(self):
        n_dofs = self.sdomain.n_dofs
        return zeros(n_dofs, 'float_')

    def setup(self, sctx):
        print('DEPRECATED CALL TO SETUP')

    def get_state_array_size(self):
        return 0

    def apply_constraints(self, K):
        for dots_eval in self.dots_list:
            dots_eval.apply_constraints(K)

    def get_corr_pred(self, U, d_U, tn, tn1, F_int,
                      step_flag='predictor', update=False, *args, **kw):

        K = self.K
        K.reset()

        U = self.tstepper.U_k
        d_U = self.tstepper.d_U
        for dots_eval in self.dots_list:
            K_mtx_arr = dots_eval.get_corr_pred(U, d_U, tn, tn1,
                                                F_int,
                                                step_flag,
                                                update, *args, **kw)
            K.sys_mtx_arrays.append(K_mtx_arr)
        return self.K

    rte_dict = Property(Dict, depends_on='dots_list')

    @cached_property
    def _get_rte_dict(self):
        rte_dict = {}

        dots_rte_dicts = []
        rte_keys = []
        for dots_eval in self.dots_list:
            dots_rte_dict = {}
            for key, eval in list(dots_eval.rte_dict.items()):
                # add the mapping here
                #
                if key not in rte_keys:
                    rte_keys.append(key)
                dots_rte_dict[key] = eval
            dots_rte_dicts.append(dots_rte_dict)

        # Get the union of all available rte keys
        for key in rte_keys:
            rte_list = []
            for rte_dict in dots_rte_dicts:
                rte_list.append(rte_dict.get(key, None))
            rte_dict[key] = tuple(rte_list)

        return rte_dict

    traits_view = View()
