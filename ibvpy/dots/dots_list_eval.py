
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, Interface, implements, \
    Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
    on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property, Dict

from traitsui.api import \
    Item, View

from traitsui.menu import \
    OKButton, CancelButton

from numpy import \
    zeros, float_, ix_, meshgrid

from ibvpy.core.i_tstepper_eval import \
    ITStepperEval
from ibvpy.core.tstepper_eval import \
    TStepperEval

from ibvpy.core.rtrace_eval import RTraceEval
from ibvpy.fets.fets_eval import IFETSEval
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
from dots_eval import DOTSEval

from time import time

#-----------------------------------------------------------------------------
# Integrator for a simple 1D domain.
#-----------------------------------------------------------------------------


class DOTSListEval(TStepperEval):

    '''
    Domain with uniform FE-time-step-eval.
    '''
    implements(ITStepperEval)

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
        print 'DEPRECATED CALL TO SETUP'

    def get_state_array_size(self):
        return 0

    def apply_constraints(self, K):
        for dots_eval in self.dots_list:
            dots_eval.apply_constraints(K)

    def get_corr_pred(self, sctx, U, d_U, tn, tn1, *args, **kw):

        K = self.K
        K.reset()

        F_int = self.F_int
        F_int[:] = 0.0

        U = self.tstepper.U_k
        d_U = self.tstepper.d_U

        for dots_eval in self.dots_list:

            K_mtx_arr = dots_eval.get_corr_pred(sctx, U, d_U, tn, tn1,
                                                self.F_int, *args, **kw)

            K.sys_mtx_arrays.append(K_mtx_arr)

        return self.F_int, self.K

    rte_dict = Property(Dict, depends_on='dots_list')

    @cached_property
    def _get_rte_dict(self):
        rte_dict = {}

        dots_rte_dicts = []
        rte_keys = []
        for dots_eval in self.dots_list:
            dots_rte_dict = {}
            for key, eval in dots_eval.rte_dict.items():
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

if __name__ == '__main__':

    from ibvpy.api import \
        TStepper as TS, RTraceGraph, RTraceDomainField, TLoop, \
        TLine, BCDof, IBVPSolve as IS, DOTSEval
    from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
    from ibvpy.mesh.fe_grid import FEGrid
    from ibvpy.mesh.fe_domain_list import FEDomainList
    from ibvpy.fets.fets1D.fets1D2l import FETS1D2L

    fets_eval = FETS1D2L(mats_eval=MATS1DElastic(E=10., A=1.))

    # Discretization
    fe_domain1 = FEGrid(coord_max=(3., 0., 0.),
                        shape = (3, ),
                        fets_eval = fets_eval)

    fe_domain2 = FEGrid(coord_min=(3., 0., 0.),
                        coord_max = (6., 0., 0.),
                        shape = (3, ),
                        fets_eval = fets_eval)

    fe_domain = FEDomainList(subdomains=[fe_domain1, fe_domain2])

    ts = TS(dof_resultants=True,
            sdomain=fe_domain,
            bcond_list=[BCDof(var='u', dof=0, value=0.),
                        BCDof(var='u', dof=4, link_dofs=[3], link_coeffs=[1.],
                              value=0.),
                        BCDof(var='f', dof=7, value=1,
                              link_dofs=[2], link_coeffs=[2])],
            rtrace_list=[RTraceGraph(name='Fi,right over u_right (iteration)',
                                     var_y='F_int', idx_y=0,
                                     var_x='U_k', idx_x=1),
                         #                        RTraceDomainField(name = 'Stress' ,
                         #                             var = 'sig_app', idx = 0),
                         #                         RTraceDomainField(name = 'Displacement' ,
                         #                                        var = 'u', idx = 0),
                         #                                 RTraceDomainField(name = 'N0' ,
                         #                                              var = 'N_mtx', idx = 0,
                         # record_on = 'update')

                         ]
            )

    # Add the time-loop control
    tloop = TLoop(tstepper=ts,
                  tline=TLine(min=0.0, step=1, max=1.0))

    print tloop.eval()
    print ts.F_int
    print 'resulting force'
    print ts.rtrace_list[0].trace.ydata
