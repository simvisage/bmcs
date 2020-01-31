import os

from ibvpy.core.i_sdomain import \
    ISDomain
from ibvpy.core.sdomain import \
    SDomain
from ibvpy.rtrace.rt_domain import RTraceDomain
from traits.api import \
    Array, Bool, Enum, Float, HasTraits, HasStrictTraits, \
    Instance, Int, Trait, Str, Enum, \
    Callable, List, TraitDict, Any, Range, \
    Delegate, Event, on_trait_change, Button, \
    Interface, WeakRef, Property, cached_property, Tuple, \
    Dict, TraitError
from traitsui.api import \
    View, Item, HSplit, VSplit
from tvtk.api import \
    tvtk

from ibvpy.core.rtrace \
    import RTrace
from ibvpy.plugins.mayavi_util.pipelines \
    import MVUnstructuredGrid, MVPolyData
from numpy \
    import ix_, mgrid, array, arange, c_, newaxis, setdiff1d, zeros, \
    float_, repeat, hstack, ndarray, append
from traitsui.api \
    import Item, View, HGroup, ListEditor, VGroup, \
    HSplit, Group, Handler, VSplit, TableEditor, ListEditor
from traitsui.menu \
    import NoButtons, OKButton, CancelButton, \
    Action
from traitsui.table_column \
    import ObjectColumn, ExpressionColumn
from traitsui.table_filter \
    import TableFilter, RuleTableFilter, RuleFilterTemplate, \
    MenuFilterTemplate, EvalFilterTemplate, EvalTableFilter
from traitsui.ui_editors.array_view_editor \
    import ArrayViewEditor


# tvtk related imports
#
class RTraceDomainInteg(RTraceDomain):

    '''
    Trace encompassing the whole spatial domain.
    '''

    fets_eval = Property

    def _get_fets_eval(self):
        return self.sd.fets_eval

    var_eval = Property

    def _get_var_eval(self):
        return self.sd.dots.rte_dict.get(self.var, None)

    def bind(self):
        '''
        Locate the evaluators
        '''
        pass

    def setup(self):
        '''
        Setup the spatial domain of the tracer
        '''
        if self.var_eval == None:
            self.skip_domain = True

    integ_val = Array(desc='Integral over the domain')

    def add_current_values(self, sctx, U_k, *args, **kw):
        '''
        Invoke the evaluators in the current context for the specified control vector U_k.
        '''
        if self.var_eval == None:
            return
        # Get the domain points
        # TODO - make this more compact. The element list is assumed to be uniform
        # so that all element arrays have the same shape. Thus, use slices and vectorized
        # evaluation to improve the performance
        sd = self.sd
        sctx.fets_eval = self.fets_eval
        field = []
        dim_slice = self.fets_eval.dim_slice
        e_arr_size = self.fets_eval.get_state_array_size()
        state_array = self.sd.dots.state_array

        # setup the result array
        integ_val = zeros((1,), dtype='float_')

        for e_id, e in enumerate(sd.elements):

            sctx.elem_state_array = state_array[e_id * e_arr_size:
                                                (e_id + 1) * e_arr_size]
            sctx.X = e.get_X_mtx()
            sctx.x = e.get_x_mtx()
            sctx.elem = e
            sctx.e_id = e_id
            field_entry = []
            i = 0
            for ip, iw in zip(self.fets_eval.ip_coords,
                              self.fets_eval.ip_weights):
                m_arr_size = self.fets_eval.m_arr_size
                sctx.mats_state_array = sctx.elem_state_array\
                    [i * m_arr_size: (i + 1) * m_arr_size]
                sctx.loc = ip
                sctx.r_pnt = ip
                sctx.p_id = i  # TODO:check this
                J_det = self.fets_eval.get_J_det(sctx.r_pnt, sctx.X)
                si = self.var_eval(sctx, U_k, *args, **kw)
                iv = si * iw * J_det
                integ_val += iv
                i += 1

            self.integ_val = integ_val

    view = View(HSplit(VSplit(VGroup('var', 'idx'),
                              VGroup('record_on', 'clear_on'),
                              Item('integ_val', style='readonly',
                                   show_label=False),
                              ),
                       ),
                resizable=True)
