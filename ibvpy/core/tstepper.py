
from ibvpy.mesh.fe_domain import FEDomain
from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.mesh.fe_refinement_grid import FERefinementGrid
from ibvpy.mesh.fe_subdomain import FESubDomain
from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
from traits.api import \
    Array, Bool, Enum, Float, HasTraits, \
    Instance, Int, Trait, Str, Enum, \
    Callable, List, Any, Range, \
    Delegate, Event, on_trait_change, Button, \
    Interface, Property, cached_property, WeakRef, Dict
from traitsui.api import \
    Item, View, HGroup, VGroup, \
    HSplit, Group, Handler, VSplit
from traitsui.menu import \
    NoButtons, OKButton, CancelButton, \
    Action
from view.ui import BMCSTreeNode

from .bcond_mngr import BCondMngr
from .i_sdomain import ISDomain
from .ibv_resource import IBVResource
import numpy as np
from .rtrace_mngr import RTraceMngr
from .scontext import SContext
from .sdomain import SDomain
from .tstepper_eval import ITStepperEval


class TStepper(BMCSTreeNode):

    """
    The TStepper is a spatially bounded TStepperEval.

    The binding is done by associating the time-stepper with a spatial
    object. In fact, any TStepper, not only the top-level one must
    be associated with spatial object. For the chained
    sub-time-steppers, this association is done using a parameter sctx
    (see the specification above). This parameters stands for spatial
    context. Note, the distinction between the spatial object and
    spatial context. While the spatial object represents a single
    spatial entity (one of domain, element, layer or material point)
    the spatial context represents a complex reference within the
    spatial object In particular, spatial context of a particular
    material point is represented as tuple containing tuple of
    references to [domain, element, layer, integration cell, material
    point].
    """

    tree_node_list = List

    def _tree_node_list_default(self):
        return [self.bcond_mngr, self.rtrace_mngr]

    # Integration terms involved in the evaluation of the
    # incremetal spatial integrals.
    # Must be either an instance of ts_eval or a tuple specifying a pair
    # ( ts_eval, sdomain ) or a list of tuples with the pairs
    # [ ( ts_eval, sdomain ), ( ts_eval, sdomain ), ... ]
    #

    # Sub-time-stepper or integrator.
    #
    tse = Instance(ITStepperEval)

    tse_integ = Property(depends_on='tse,_sdomain, _sdomain.changed_structure')

    @cached_property
    def _get_tse_integ(self):
        if self.tse:
            self.tse.tstepper = self
            return self.tse
        else:
            self.sdomain.dots.tstepper = self
            return self.sdomain.dots

    # Spatial domain to bind the time-stepper to.
    # For convenience automatically convert the plain list to FEDomainList
    #
    _sdomain = Instance(ISDomain)
    sdomain = Property(Instance(ISDomain))

    def _set_sdomain(self, value):
        if isinstance(value, FEGrid):
            # construct FERefinementGrid and FEDomain
            self._sdomain = FEDomain()
            fe_rgrid = FERefinementGrid(domain=self._sdomain,
                                        fets_eval=value.fets_eval)
            value.level = fe_rgrid
        elif isinstance(value, FERefinementGrid):
            # construct FEDomain
            self._sdomain = FEDomain()
            value.domain = self._sdomain
        elif isinstance(value, list):
            self._sdomain = FEDomain()
            for d in value:
                if isinstance(d, FEGrid):
                    fe_rgrid = FERefinementGrid(domain=self._sdomain,
                                                fets_eval=d.fets_eval)
                    d.level = fe_rgrid
                elif isinstance(d, FESubDomain):
                    d.domain = self._sdomain
                else:
                    raise TypeError('The list can contain only FEGrid or FERefinementGrid')
        else:
            self._sdomain = value

    def _get_sdomain(self):
        if self._sdomain == None:
            self._sdomain = SDomain()
        return self._sdomain

    subdomains = Property()

    def _get_subdomains(self):
        if self.sdomain == None:
            return []
        return self.sdomain.subdomains

    xdomains = Property()

    def _get_xdomains(self):
        if self.sdomain == None:
            return []
        return self.sdomain.xdomains

    def redraw(self):
        self.sdomain.redraw()

    sctx = Instance(SContext)

    # Boundary condition manager
    #
    bcond_mngr = Instance(BCondMngr)

    def _bcond_mngr_default(self):
        return BCondMngr()

    # Convenience constructor
    #
    # This property provides the possibility to write
    # tstepper.bcond_list = [BCDof(var='u',dof=5,value=0, ... ]
    # The result gets propageted to the BCondMngr
    #
    bcond_list = Property(List)

    def _get_bcond_list(self):
        return self.bcond_mngr.bcond_list

    def _set_bcond_list(self, bcond_list):
        self.bcond_mngr.bcond_list = bcond_list

    # Response variable manager
    #
    rtrace_mngr = Instance(RTraceMngr)

    def _rtrace_mngr_default(self):
        return RTraceMngr(tstepper=self)

    # Convenience constructor
    #
    # This property provides the possibility to write
    # tstepper.bcond_list = [RVDof(var='u',dof=5,value=0, ... ]
    # The result gets propageted to the RTraceMngr
    #
    rtrace_list = Property(List)

    def _get_rtrace_list(self):
        return self.rtrace_mngr.rtrace_list

    def _set_rtrace_list(self, rtrace_list):
        self.rtrace_mngr.rtrace_list = rtrace_list

    # Possibility to add a callable for derived
    # variables of the control variable.
    u_processor = Callable

    # Backward reference to the time-loop in order to accommadate the
    # response-trace-evaluators from the top level. These include
    # bookkeeping data like memory usage or solving time per selected
    # type of operation.
    #
    tloop = WeakRef

    dir = Property

    def _get_dir(self):
        return self.tloop.dir

    dof_resultants = Bool(True)

    rte_dict = Property(Dict, depends_on='tse')

    @cached_property
    def _get_rte_dict(self):
        '''
        Gather all the currently applicable evaluators from the sub-ts
        and from the time-loop.

        Note the control data (time-loop data) is available within the
        model to construct flexible views (tracers) on the model.
        '''
        _rte_dict = {}

        def _get_F_int(sctx, U_k, *args, **kw):
            return self.F_int

        if self.dof_resultants:
            _rte_dict['F_int'] = _get_F_int  # lambda sctx, U_k: self.F_int
            _rte_dict['F_ext'] = lambda sctx, U_k, *args, **kw: self.F_ext

        _rte_dict.update(self.tse_integ.rte_dict)
        if self.tloop:
            _rte_dict.update(self.tloop.rte_dict)
        return _rte_dict

    def new_cntl_var(self):
        return self.tse_integ.new_cntl_var()

    def new_resp_var(self):
        return self.tse_integ.new_resp_var()

    kw = Dict
    args = List
    K = Instance(SysMtxAssembly)

    U_k = Property(depends_on='_sdomain.changed_structure')

    @cached_property
    def _get_U_k(self):
        '''
         Setup the primary state variables on demand
        '''
        U_k = self.new_cntl_var()
        return U_k

    d_U = Property(depends_on='_sdomain.changed_structure')

    @cached_property
    def _get_d_U(self):
        '''
         Current increment of the displacement variable
        '''
        return self.new_cntl_var()

    F_ext = Property(depends_on='_sdomain.changed_structure')

    @cached_property
    def _get_F_ext(self):
        '''
         Return the response variable to be used when assembling the
         boundary conditions. Should the bcond_mngr take care of this?
         That's the source object, isn't it? BCondMngr is the bounded
         version of the conditions, it could supply the matrix
         autonomously.

        '''
        return self.new_resp_var()

    update_state_on = Bool(False)
    # missing - setup of the time-stepper itself. reacting to changes
    # in the sub time-steppers. bcond_list and rtrace_list must be reset once
    # a change has been performed either in a spatial domain or in
    # tse.
    #

    def setup(self):

        # Put the spatial domain into the spatial context
        #
        self.sctx = sctx = self.sdomain.new_scontext()
        self.sctx.sdomain = self.sdomain

        # Let the boundary conditions setup themselves within the
        # spatial context
        #
        # TODO - the setup needs the link to the algorithm and to the
        # time-steppers as well.!
        #
        self.bcond_mngr.setup(sctx)

        # Let the response variables setup themselves within the
        # spatial context
        #
        self.rtrace_mngr.setup(self.sdomain)

        # Set up the system matrix
        #
        self.K = SysMtxAssembly()

        # Register the essential boundary conditions in the system matrix
        #
        self.bcond_mngr.apply_essential(self.K)

        self.tse_integ.apply_constraints(self.K)

        # Prepare the global update flag
        self.update_state_on = False

        if self.u_processor:
            if self.u_processor.sd == None:
                self.u_processor.sd = self.sdomain

        # add-on parameters to be passed to every inner loop
        # they are provided by u_processors
        #
        self.kw = {}
        self.args = []

    F_int = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_F_int(self):
        n_dofs = self.sdomain.n_dofs
        return np.zeros(n_dofs, 'float_')

    def eval(self, U_k, d_U, t_n, t_n1,
             step_flag='predictor'):
        '''Get the tangential operator (system matrix) and residuum
        associated with the current time step.

        @param step_flag: indicator of the predictor | corrector step
           it is needed for proper handling of constraint equations
           (essential boundary conditions and explicit links between
           the variables.

        @param U_k: current value of the control variable (including
            the value of the last increment d_U

        @param d_U: increment of the control variable
            U[k] = U[k-1] + d_U

        @param t_n: value of the time control parameters in the
            last equilibrated step.

        @param t_n1: value of the target time in the current
            time step.
        '''

        # Reset the system matrix (constraints are preserved)
        #
        self.K.reset_mtx()
        self.F_int[:] = 0.0

        # Put the spatial domain into the spatial context
        #
        sctx = self.sctx

        if self.u_processor:
            self.args, self.kw = self.u_processor(U_k)

        # Let the time sub-stepper evaluate its contribution.
        #
        K_mtx = self.tse_integ.get_corr_pred(U_k, d_U, t_n, t_n1,
                                             self.F_int,
                                             step_flag,
                                             self.update_state_on,
                                             *self.args, **self.kw)

        # Promote the system matrix to the SysMtxAssembly
        # Supported representation of the system matrix is
        # float, ndarray, SysMtxArray and SysMtxAssembly
        #
        # @todo use coerce in order to hide this conversions.
        # or is adapter concept of traits a possibility?
        #
        if isinstance(K_mtx, np.ndarray):
            self.K.add_mtx(K_mtx)
        elif isinstance(K_mtx, SysMtxArray):
            self.K.sys_mtx_arrays.append(K_mtx)
        elif isinstance(K_mtx, list):
            self.K.sys_mtx_arrays = K_mtx
        elif isinstance(K_mtx, SysMtxAssembly):
            self.K.sys_mtx_arrays = K_mtx.sys_mtx_arrays

        norm_F_int = np.linalg.norm(self.F_int)
        # Switch off the global update flag
        #
        self.update_state_on = False

        # Apply the boundary conditions
        #
        if self.dof_resultants == True:

            # Prepare F_ext by zeroing it
            #
            self.F_ext[:] = 0.0

            # Assemble boundary conditions in K and self.F_ext
            #
            self.bcond_mngr.apply(
                step_flag, sctx, self.K, self.F_ext, t_n, t_n1)

            # Return the system matrix assembly K and the residuum
            #
            return self.K, self.F_ext - self.F_int, norm_F_int

        else:

            # On the other hand, the time-loop only requires the residuum
            # which can be obtained without an additional
            # memory consumption by issuing an in-place switch of the sign
            #
            self.F_int *= -1  # in-place sign change of the internal forces
            #
            # Then F_int can be used as the target for the boundary conditions
            #
            self.bcond_mngr.apply(
                step_flag, sctx, self.K, self.F_int, t_n, t_n1)

            #
            # The subtraction F_ext - F_int has then been performed implicitly
            # and the residuum can be returned by issuing
            #
            return self.K, self.F_int, norm_F_int

    def update_state(self, U):
        '''
        spatial context represents a stack with the top object
         representing the current level.
        @param U:
        '''
        #sctx = ( self.sdomain, )
        self.update_state_on = True
        #self.tse_integ.update_state( sctx, U )

    def register_mv_pipelines(self, e):
        '''Register the visualization pipelines in mayavi engine
        '''
        self.tse_integ.register_mv_pipelines(e)
        scene = e.new_scene()
        scene.name = 'Spatial domain'
        self.sdomain.register_mv_pipelines(e)
        self.rtrace_mngr.register_mv_pipelines(e)

    traits_view = View(Group(Item('sdomain', style='custom', show_label=False),
                             label='Discretization'),
                       Group(Item('tse', style='simple', show_label=False),
                             label='Integrator'),
                       Group(Item('bcond_mngr', style='custom', show_label=False),
                             label='Boundary conditions'),
                       Group(Item('dof_resultants'),
                             label='Options'),
                       resizable=True,
                       height=0.8,
                       width=0.8,
                       buttons=[OKButton, CancelButton],
                       kind='subpanel',
                       )
