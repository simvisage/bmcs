import os

from ibvpy.api import RTrace
from ibvpy.core.i_sdomain import \
    ISDomain
from ibvpy.core.sdomain import \
    SDomain
from ibvpy.plugins.mayavi_util.pipelines import \
    MVUnstructuredGrid, MVPolyData
from numpy import ix_, mgrid, array, arange, c_, newaxis, setdiff1d, zeros, \
    float_, vstack, hstack, repeat
from traits.api import \
    Array, Bool, Enum, Float, HasTraits, HasStrictTraits, \
    Instance, Int, Trait, Str, Enum, \
    Callable, List, TraitDict, Any, Range, \
    Delegate, Event, on_trait_change, Button, \
    Interface, WeakRef, Property, cached_property, Tuple, \
    Dict, Any
from traitsui.api import Item, View, HGroup, ListEditor, VGroup, \
    HSplit, Group, Handler, VSplit, TableEditor, ListEditor
from traitsui.api import View, Item, HSplit, VSplit
from traitsui.menu import NoButtons, OKButton, CancelButton, \
    Action
from tvtk.api import tvtk
from tvtk.api import tvtk

from traitsui.table_column \
    import ObjectColumn, ExpressionColumn
from traitsui.table_filter \
    import TableFilter, RuleTableFilter, RuleFilterTemplate, \
    MenuFilterTemplate, EvalFilterTemplate, EvalTableFilter
from traitsui.ui_editors.array_view_editor \
    import ArrayViewEditor

from .rt_domain_integ import RTraceDomainInteg
from .rt_domain_list import RTraceDomainList


# tvtk related imports
#
class RTraceDomainListInteg(RTrace, RTraceDomainList):

    #    sd = Instance( SDomain )
    #
    #    rt_domain = Property
    #    def _get_rt_domain(self):
    #        return self.sd.rt_bg_domain

    label = Str('RTraceDomainInteg')
    var = Str('')
    idx = Int(-1, enter_set=True, auto_set=False)

    save_on = Enum('update', 'iteration')
    warp = Bool(False)
    warp_f = Float(1.)
    warp_var = Str('u')

    def bind(self):
        '''
        Locate the evaluators
        '''

    def setup(self):
        '''
        Setup the spatial domain of the tracer
        '''
        for sf in self.subfields:
            sf.setup()

    subfields = Property

    @cached_property
    def _get_subfields(self):
        # construct the RTraceDomainFields
        #
        return [RTraceDomainInteg(var=self.var,
                                  idx=self.idx,
                                  position=self.position,
                                  save_on=self.save_on,
                                  warp=self.warp,
                                  warp_f=self.warp_f,
                                  sd=subdomain)
                for subdomain in self.sd.nonempty_subdomains]

    integ_val = Array(desc='Integral over the domain')

    def add_current_values(self, sctx, U_k, *args, **kw):
        integ_val = array([0.0], 'float_')
        for sf in self.subfields:
            if sf.skip_domain:
                continue
            sf.add_current_values(sctx, U_k, *args, **kw)
            integ_val += sf.integ_val
        self.integ_val = integ_val

    def timer_tick(self, e):
        pass

    def write(self):
        pass

    def clear(self):
        pass

    view = View(HSplit(VSplit(VGroup('var', 'idx'),
                              VGroup('record_on', 'clear_on'),
                              Item('integ_val', style='readonly',
                                   show_label=False),
                              ),
                       ),
                resizable=True)
