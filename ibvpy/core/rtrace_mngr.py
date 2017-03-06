
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, \
    Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
    Dict, Property, cached_property, WeakRef, Delegate, Button, \
    Constant

from traitsui.api import \
    Item, View, HGroup, ListEditor, VGroup, VSplit, Group, HSplit, \
    TabularEditor

from traitsui.menu import \
    NoButtons, OKButton, CancelButton, Action, CloseAction, Menu, \
    MenuBar, Separator

from traitsui.tabular_adapter \
    import TabularAdapter

from numpy import zeros, float_
from ibv_resource import IBVResource
from rtrace import RTrace

#-------------------------------------------------------------------------
# Tabular Adapter Definition
#-------------------------------------------------------------------------


class RTraceTableAdapter (TabularAdapter):

    columns = [('Name', 'name'),
               #                ( 'Record on',    'record_on' ),
               #                ( 'Clear on',     'clear_on' )
               ]

    font = 'Courier 10'
    variable_alignment = Constant('right')

#-------------------------------------------------------------------------
# Tabular Editor Construction
#-------------------------------------------------------------------------
rtrace_editor = TabularEditor(
    selected='current_rtrace',
    adapter=RTraceTableAdapter(),
    operations=['move'],
    auto_update=True
)


class RTraceMngr(IBVResource):

    '''
    The response variable manager.

    The RTraceMngr can be associated with a data . It looks for an
    attribute rtracears = [RTrace( name = "var1", ... ), ... ]
    and locates the specification of these variables

    The manager gathers all the RTrace specifications included in the
    object o. The RTraces can then be applied in ResponseView
    specification (RView) in different spatial contexts. It may be
    either associated to a particlar axis in a single-point plot or as
    a y-axis in the LinePlot. Further it can be specified as a field
    variable in an 2D or 3D iso-plot.
    '''

    # service specifiers - used to link the service to this object
    service_class = 'ibvpy.plugins.rtrace_service.RTraceService'
    service_attrib = 'rtrace_mngr'

    # Traced object an object suplying the RT Evaluators
    #
    tstepper = WeakRef

    dir = Property

    def _get_dir(self):
        return self.tstepper.dir

    # List of response evaluators available in the current time stepper
    #
    rte_dict = Property(Dict, depends_on='tstepper')

    def _get_rte_dict(self):
        return self.tstepper.rte_dict

    # List of response tracers specified in the input
    #
    rtrace_list = List(RTrace)
    rtrace_bound_list = Property(List(RTrace), depends_on='rtrace_list')

    @cached_property
    def _get_rtrace_bound_list(self):
        '''
        Access the rtrace list after setting a backward reference
        from the individual traces.
        '''
        for rtrace in self.rtrace_list:
            rtrace.rmgr = self
        return self.rtrace_list

    # variable selectable in the table of varied params (just for viewing)
    current_rtrace = Instance(RTrace)

    def _current_rtrace_default(self):
        if len(self.rtrace_list) > 0:
            return self.rtrace_list[0]

    timer = Instance(Any)

    def __init__(self, **kwtraits):
        super(RTraceMngr, self).__init__(**kwtraits)
        self.timer = None

    def setup(self, sd):
        # self.clear()
        if self.tstepper:
            for rtrace in self.rtrace_bound_list:
                rtrace.bind()
        for rtrace in self.rtrace_bound_list:
            rtrace.sd = sd
            rtrace.setup()

    def start_timer(self):
        if self.timer:
            self.timer.Start(1000.0, wx.TIMER_CONTINUOUS)

    def stop_timer(self):
        if self.timer:
            self.timer.Stop()

    def timer_tick(self, e=None):
        for rte in self.get_values():
            rte.timer_tick(e)

    def get_values(self):
        return self.rtrace_bound_list

    def record(self, sctx, U_k, *args, **kw):
        for rte in self.get_values():
            rte.add_current_values(sctx, U_k,
                                   *self.tstepper.args, **self.tstepper.kw)
            rte.add_current_displ(sctx, U_k)

    def record_iter(self, sctx, U_k, *args, **kw):
        for rte in self.get_values():
            if rte.record_on == 'iteration':
                rte.add_current_values(sctx, U_k,
                                       *self.tstepper.args, **self.tstepper.kw)
                rte.add_current_displ(sctx, U_k)

    def record_equilibrium(self, sctx, U_k):
        for rte in self.get_values():
            if rte.record_on == 'update':
                rte.add_current_values(sctx, U_k,
                                       *self.tstepper.args, **self.tstepper.kw)
                rte.add_current_displ(sctx, U_k)
            if rte.save_on == 'update':
                rte.write()
            if rte.clear_on == 'update':
                rte.clear()

    def close(self):
        for rte in self.get_values():
            rte.close()

    def clear(self, e=None):
        for rte in self.get_values():
            rte.clear()

    def __getitem__(self, name):
        for rtrace in self.rtrace_bound_list:
            if rtrace.name == name:
                return rtrace
        return IndexError, 'rtrace %s not found in rtrace_mngr' % name

    def register_mv_pipelines(self, e):
        '''Register the visualization pipelines in mayavi engine

        The method runs over the response traces and lets them add their pipeline
        components into the mayavi engine.
        '''
        for rtrace in self.rtrace_bound_list:
            rtrace.register_mv_pipelines(e)

#    warp_var = Str( 'u' )
#    warp_field = Property( Array )
#    def _get_warp_field(self):
#        ''' Search for the field tracer with warpable data and use it as a warp field '''
#        for rtrace in self.rtrace_bound_list:
#            if rtrace.label == 'RTraceDomainField' and rtrace.var == self.warp_var:
#                warp_field = zeros((rtrace.field_arr.shape[0], 3), float_)
#                warp_field[:,rtrace.var_eval.dim_slice] = rtrace.field_arr
#                return warp_field
#
# raise KeyError, 'warp field not found'

    trait_view = View(HSplit(
        Item('rtrace_bound_list', show_label=False,
             editor=rtrace_editor),
        Item('current_rtrace', show_label=False,
             style='custom', resizable=True),
    ),
        resizable=True,
        scrollable=True,
        height=0.6, width=0.6
    )

if __name__ == '__main__':
    from rtrace import RTraceGraph
    rmgr = RTraceMngr(rtrace_list=[
        RTraceGraph(name='rte x'),
        RTraceGraph(name='rte 2'),
        RTraceGraph(name='rte 3'),
        RTraceGraph(name='rte 4'),
        RTraceGraph(name='rte 5'),
        RTraceGraph(name='rte 6'),
        RTraceGraph(name='rte 7'),
        RTraceGraph(name='rte 8'),
        RTraceGraph(name='rte 8'),
        RTraceGraph(name='rte 10'),
        RTraceGraph(name='rte 11'),
    ])
    rmgr.configure_traits()
