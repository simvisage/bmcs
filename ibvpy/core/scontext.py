from traits.api import \
    Array, Bool, Enum, Float, HasTraits, HasStrictTraits, \
    Instance, Int, Trait, Str, Enum, \
    Callable, List, TraitDict, Any, Range, \
    Delegate, Event, on_trait_change, Button, \
    Interface, WeakRef, implements, Property, cached_property, Tuple, \
    Dict
from traitsui.api import Item, View, HGroup, ListEditor, VGroup, \
    HSplit, Group, Handler, VSplit, TableEditor, ListEditor

from traitsui.menu import NoButtons, OKButton, CancelButton, \
    Action

from traitsui.ui_editors.array_view_editor \
    import ArrayViewEditor

from traitsui.table_column \
    import ObjectColumn, ExpressionColumn

from traitsui.table_filter \
    import TableFilter, RuleTableFilter, RuleFilterTemplate, \
    MenuFilterTemplate, EvalFilterTemplate, EvalTableFilter

from numpy import linspace, ix_, mgrid, ogrid, array, hstack, vstack, zeros, arange, c_, newaxis

# tvtk related imports
#
# from etsproxy.tvtk.pyface.actor_model import ITVTKActorModel
# from etsproxy.tvtk.pyface.actor_editor import ActorEditor
# from etsproxy.tvtk.pyface import actors
# from etsproxy.tvtk.api import tvtk


class SContext:  # (HasTraits):

    '''
    Spatial context represents a complex reference within the
    spatial object.

    In particular, spatial context of a particular material point is
    represented as tuple containing tuple of references to [domain,
    element, layer, integration cell, material point]

    The context is filled when stepping over the discretization
    levels. It is included in all parameters of the time-step-evals
    and resp-trace-evals.
    '''

    # Subsidiary integration mapping arrays - shouldn't it be part of the context array?
    # The eval should get a chance to put the integration mappings into the spatial
    # context object. In order to make the access faster afterwards.
    #
#    elcoord_array = Property( Array, depends_on = 'sdomain:elements')
#    @cached_property
#    def _get_elcoord_array(self):
#
#        shape = self.sdomain.shape
#        n_e_nodes = self.fets_eval.n_e_nodes
#        elcoord_array = zeros(  shape * n_e_nodes * 3 ).reshape(shape,n_e_nodes,3)
#        i = 0
#        for e in self.elements:
#            elcoord_array[i,:,:] = e.get_X_mtx()
#            i += 1
#        return elcoord_array
