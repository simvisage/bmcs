from enthought.traits.api import \
    Array, Bool, Enum, Float, HasTraits, \
    Instance, Int, Trait, Str, Enum, \
    Callable, List, TraitDict, Range, \
    Delegate, Event, on_trait_change, Button, \
    Interface, implements, Property, cached_property

from enthought.traits.ui.api import \
    Item, View, HGroup, ListEditor, VGroup, \
    HSplit, Group, Handler, VSplit

from enthought.traits.ui.api \
    import View, Item, VSplit, TableEditor, ListEditor

from enthought.traits.ui.table_filter \
    import TableFilter, RuleTableFilter, RuleFilterTemplate, \
           MenuFilterTemplate, EvalFilterTemplate, EvalTableFilter

from enthought.traits.ui.table_column \
    import ObjectColumn, ExpressionColumn

from enthought.traits.ui.table_filter \
    import TableFilter, RuleTableFilter, RuleFilterTemplate, \
           MenuFilterTemplate, EvalFilterTemplate, EvalTableFilter

from numpy import ix_, array, int_, dot, newaxis, float_, copy
from i_bcond import IBCond

# The definition of the demo TableEditor:
bcond_list_editor = TableEditor( 
    columns = [ ObjectColumn( label = 'Type', name = 'var' ),
                ObjectColumn( label = 'Value', name = 'value' ),
                ObjectColumn( label = 'DOF', name = 'dof' )
                ],
    editable = False,
    selected = 'object.selected_bcond',
    )

class BCondMngr( HasTraits ):

    bcond_list = List( IBCond )

    selected_bcond = Instance( IBCond )

    def setup( self, sctx ):
        '''
        '''
        for bc in self.bcond_list:
            bc.setup( sctx )

    def apply_essential( self, K ):
        '''Register the boundary condition in the equation system.
        '''
        for bcond in self.bcond_list:
            bcond.apply_essential( K )

    def apply( self, step_flag, sctx, K, R, t_n, t_n1 ):

        for bcond in self.bcond_list:
            bcond.apply( step_flag, sctx, K, R, t_n, t_n1 )

    traits_view = View( VSplit( Item( 'bcond_list', style = 'custom', editor = bcond_list_editor,
                                     show_label = False ),
                                Item( 'selected_bcond@', show_label = False ) ),
                        resizable = True,
                        kind = 'subpanel',
                        )
