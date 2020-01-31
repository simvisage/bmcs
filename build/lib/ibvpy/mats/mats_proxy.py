
# Traits imports
from traits.api import \
    Enum, Float, \
    Instance, Trait, HasTraits, on_trait_change, \
    Dict, Property, cached_property, WeakRef, String, \
    Constant, List
from traitsui.api import \
    Item, View, HSplit, Group, TabularEditor

from ibvpy.api import BCDof
from ibvpy.api import RTDofGraph
from ibvpy.core.ibvp_solve import IBVPSolve as IS
from ibvpy.core.tloop import TLoop, TLine
from ibvpy.core.tstepper import \
    TStepper as TS
from ibvpy.mats.mats_eval import IMATSEval, MATSEval
from mathkit.mfn.mfn_ndgrid.mfn_ndgrid import MFnNDGrid, GridPoint
from traitsui.tabular_adapter \
    import TabularAdapter


# Traits UI imports
# Numpy imports
#-------------------------------------------------------------------------
#                                     VariedParam
#-------------------------------------------------------------------------
class VariedParam(HasTraits):
    """
    Association between the spatial function and material parameter.
    """
    mats_eval = WeakRef(IMATSEval)

    varname = String

    reference_value = Property(Float, depends_on='mats_eval')

    @cached_property
    def _get_reference_value(self):
        return getattr(self.mats_eval, self.varname)

    switch = Enum('constant', 'varied')

    spatial_fn = Instance(MFnNDGrid)

    variable = Property(Float)

    def _get_variable(self):
        return getattr(self.mats_eval, self.varname)

    def _set_variable(self, value):
        setattr(self.mats_eval, self.varname, value)

    def adjust_for_sctx(self, sctx):
        if self.switch == 'varied':
            X_pnt = sctx.fets_eval.get_X_pnt(sctx)
            # X_coord = zeros(3)
            # X_coord[0:2] = X_pnt
            coeff = self.spatial_fn(X_pnt)
            self.variable = self.reference_value * coeff

    traits_view = View(Group(Item('varname', style='readonly', show_label=False),
                             Item('reference_value', style='readonly'),
                             Item('switch', style='custom', show_label=False),
                             Item('spatial_fn', style='custom', show_label=False, resizable=True)),
                       resizable=True,
                       height=800)

#-------------------------------------------------------------------------
# Tabular Adapter Definition
#-------------------------------------------------------------------------


class VariedParamAdapter (TabularAdapter):

    columns = [('Name', 'varname'),
               ('Variable', 'variable')]

    font = 'Courier 10'
    variable_alignment = Constant('right')


#-------------------------------------------------------------------------
# Tabular Editor Construction
#-------------------------------------------------------------------------
varpar_editor = TabularEditor(
    selected='current_varpar',
    adapter=VariedParamAdapter(),
    operations=['move'],
    auto_update=True
)

#---------------------------------------------------------------------------
# Material Proxy to include variable parameters
#---------------------------------------------------------------------------


class MATSProxy(MATSEval):
    '''
    Material model with spatially varying material parameters.

    @author: rch
    The material model works as a proxy delegating the standard
    functionality to the associated material model @param mats_eval.

    In the first step, the proxy identifies the material parameters
    of the mats_eval. The identification is performed by scanning the
    @param mats_eval for float traits using the method identify parameters.
    The scanning happens on demand when the @param
    varpars list is accessed.

    For each variable an instance of the VarPar object gets constructed
    to handle the spatial variation. By default, the VarPar has the
    switch set to constant so that the proxy has no effect.

    Spatial variation of a parameter within the VarPar instance
    can be activated both in the script during the model construction
    and interactively thorough the user interface. Both these ways are
    now briefly described to illuminate the internal workings
    of the proxy.

    Scripting interface
    ===================
    There is a quite number of dependencies involved in associating a
    spatial profile to a material parameter. Since this implementation
    is meant generic to be applicable to any material model,
    these dependcies must be captured in a transparent way. Let us
    identify the dependencies using a particular example.

    *** initial state ***

    proxy has an associated default material model.

    (@note: actually, it would be reasonable to reflect the usage
    context of the proxy, i.e. reuse the mats_eval used within of the
    element using the mats_proxy. Probably the spatial variation should intervene
    at a global level and hook up the spatial context anywhere upon the
    change of the spatial coordinate. In other words,
    the adjustment of a parameter in a particular time stepper would be done
    whenever the spatial coordinate gets changed.

    However, handling of these association
    is not the current priority. It can be solved by subclassing
    the proxy for the five supported dimensions, i.e. 1d, 1d5, 2d, 2d5 and 3d)

    upon access to varpar: identify the parameters of
    @param mats_eval and construct the VarPar instances. They are all
    set to constant.

    *** assign a spatial profile ***

    The variables must be accessible via their keywords. This means they are
    managed within a dictionary. For  instance you can issue

    mp = MATSProxy( mats_eval = MATS1DElastic() )
    mp.varpars['E'].spatial_fn.x_mins = [0,0,0]
    mp.varpars['E'].spatial_fn.x_maxs = [10,10,0]
    mp['E'].spatial_fn.shape = (500,5,0)
    mp['E'].spatial_fn.set_values_in_box( 10, [0,0,0], [10,10,8] )

    @todo The spatial variation can be handled at the level
    of the tstepper     Then, the identification of parameters
    can be performed for all integration levels. Further,
    the spatial binding can be taken into account.
    i.e., the spatial profile is specified for an existing
    domain with particular bounding box. The VaPars are registered
    in the spatial context and the adjustment
    of the material parameters is invoked dynamically,
    whenever the @param r_pnt gets changed. That means the correct
    parameters are available both for
    the iteration and for response tracing.

    The extensions needed -

    1) the recursive list of sub time steppers during the
        parameter identification in tstepper_eval

    2) spatial binding to the geometric model as a
        separate step (identification of the bounding box)

    3) spatial context is launched before starting the computation.
        at this point, the varpar_dict must be registered within
        the spatial context.

    3) spatial context must be hooked with the call to
       the adjust state variables.

    4) any change in the tstepper structure must
       cause reconstruction of the varpar list. Only those
       varpars should be modified that are really affected.
       (the need for change notification mechanism within the
       tstepper hierarchy)
    '''
#    mats_eval_type = Enum('MATS1DElastic',
#                          'MATS1DDamage',
#                          'MA2DCompositeMicroplaneDamage',
#                          'MATS2DScalarDamage')
#    mats_eval = Property( Instance( IMATSEval ), depends_on = 'mats_eval_type' )
#    @cached_property
#    def _get_mats_eval(self):
#        return eval( self.mats_eval_type + '()' )

#    def _set_mats_eval(self, value):
#        return value
    mats_eval = Instance(IMATSEval)
    #-------------------------------------------------------------------------
    # Management of spatially varying parameters depending on the value of mats_eval
    #-------------------------------------------------------------------------
    varpars = Dict

    def _varpars_default(self):
        return self._get_varpars()

    @on_trait_change('mats_eval')
    def _update_varpars(self):
        self.varpars = self._get_varpars()

    def _get_varpars(self):
        '''
        reset the varpar list according to the current mats_eval object.
        '''
        params = self.mats_eval.identify_parameters()
        varset = {}
        for key, par in list(params.items()):
            par_val = getattr(self.mats_eval, key)
            varset[key] = VariedParam(mats_eval=self.mats_eval,
                                      varname=key)
            #                                spatial_fn = 1 ) )
        return varset

    varpar_list = Property(List(VariedParam), depends_on='varpars')

    @cached_property
    def _get_varpar_list(self):
        return list(self.varpars.values())

    # variable selectable in the table of varied params (just for viewing)
    current_varpar = Instance(VariedParam)

    def _current_varpar_default(self):
        return self.varpar_list[0]

    @on_trait_change('mats_eval')
    def set_current_varpar(self):
        self.current_varpar = self.varpar_list[0]

    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------

    def get_state_array_size(self):
        return self.mats_eval.get_state_array_size()

    def setup(self, sctx):
        '''
        Intialize state variables.
        '''
        self.mats_eval.setup(sctx)

    def new_cntl_var(self):
        return self.mats_eval.new_cntl_var()

    def new_resp_var(self):
        return self.mats_eval.new_resp_var()

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1, eps_avg=None):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        #
        # Reset the parameter values for the material model according to the
        # current spatial context
        #

        #        @todo check whether or not to put this into the spatial context - the X_pnt
        #        is needed only for certain configurations. If implemented as a cached functor
        #        it could be constructed on demand for all gauss points as a cached property
        # and the material models could reuse it without repeated evalution of
        # this operation.

        for varpar in self.varpar_list:
            varpar.adjust_for_sctx(sctx)

        return self.mats_eval.get_corr_pred(sctx, eps_app_eng, d_eps, tn, tn1, eps_avg)

    #-------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # Response trace evaluators
    #-------------------------------------------------------------------------

#    def get_eps_app( self, sctx, eps_app_eng ):
#        return eps_app_eng
#
#    def get_sig_app( self, sctx, eps_app_eng ):
#        # @TODO
#        # the stress calculation is performed twice - it might be
#        # cached.
#        sig_eng, D_mtx = self.get_corr_pred( sctx, eps_app_eng, 0, 0 )
#        return sig_eng

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return self.mats_eval.rte_dict

    #-------------------------------------------------------------------------
    # View specification
    #-------------------------------------------------------------------------
    traits_view = View(Group(
        Item('mats_eval_type', show_label=False, style='custom'),
        Item('mats_eval', show_label=False, style='custom'),
        label='Material_model'
    ),
        Group(
        HSplit(
            Item('varpar_list', show_label=False, editor=varpar_editor),
            Item('current_varpar', show_label=False,
                 style='custom', resizable=True),
        ),
        label='Spatially varied parameters'
    ),
        width=0.8,
        height=0.8,
        resizable=True)


#-------------------------------------------------------------------------
# Example
#-------------------------------------------------------------------------

# from mats1D.mats1D_damage.mats_damage1d import MATS1DDamage


if __name__ == '__main__':
    # tseval for a material model
    #
    # tseval  = MATSProxy(  mats_eval_type = 'MATS1DElastic' )
    tseval = MATSProxy()
    E_mod_varpar = tseval.varpars['E']
    E_mod_varpar.spatial_fn = MFnNDGrid(shape=(10, 10, 1),
                                        x_maxs=GridPoint(x=10, y=10))

    ts = TS(tse=tseval,
            bcond_list=[BCDof(var='u', dof=0, value=1.)
                        ],
            rtrace_list=[RTDofGraph(name='strain 0 - stress 0',
                                    var_x='eps_app', idx_x=0,
                                    var_y='sig_app', idx_y=0,
                                    record_on='update')
                         ]
            )
    # Put the time-stepper into the time-loop
    #

    tmax = 4.0
    # tmax = 0.0006
    n_steps = 100

    tl = TLoop(tstepper=ts,
               DT=tmax / n_steps, KMAX=100, RESETMAX=0,
               tline=TLine(min=0.0, max=tmax))

    isim = IS(tloop=tl)
    isim.configure_traits()
