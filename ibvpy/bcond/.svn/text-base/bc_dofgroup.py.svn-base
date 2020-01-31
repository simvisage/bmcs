from enthought.traits.api import Array, Bool, Enum, Float, HasTraits, \
                                 Instance, Int, Trait, Str, Enum, \
                                 Callable, List, TraitDict, Any, Range, \
                                 Delegate, Event, on_trait_change, Button, \
                                 Interface, implements, Property, cached_property
from enthought.traits.ui.api import Item, View, HGroup, ListEditor, VGroup, \
     HSplit, Group, Handler, VSplit
from enthought.traits.ui.menu import NoButtons, OKButton, CancelButton, \
     Action
from enthought.traits.ui.api \
    import View, Item, VSplit, TableEditor, ListEditor
from enthought.traits.ui.table_column \
    import ObjectColumn, ExpressionColumn
from numpy import \
    ix_, array, int_, dot, newaxis, float_, copy, repeat
from ibvpy.api import \
    IBCond
from bc_dof import BCDof

# The definition of the demo TableEditor:
bcond_list_editor = TableEditor( 
    columns = [ ObjectColumn( label = 'Type', name = 'var' ),
                ObjectColumn( label = 'Value', name = 'value' ),
                ObjectColumn( label = 'DOF', name = 'dof' )
                ],
    editable = False,
    )

from ibvpy.plugins.mayavi.pipelines import \
    MVPolyData, MVPointLabels, MVStructuredGrid

class BCDofGroup( HasTraits ):
    '''
    Implements the IBC functionality for a constrained dof.
    '''
    implements( IBCond )

    var = Enum( 'u', 'f' )

    get_dof_method = Callable

    get_link_dof_method = Callable  # optional callable to deliver
                                    # dofs for linked pairing
    dof_numbers = Array( int )
    dof_X = Array( Float )

    bcdof_list = List( BCDof )

    # List of dofs that determine the value of the current dof
    #
    # If this list is empty, then the current dof is
    # prescribed. Otherwise, the dof value is given by the 
    # linear combination of DOFs in the list (see the example below)
    #
    link_dofs = List( Int )

    # Coefficients of the linear combination of DOFs specified in the
    # above list.
    #
    link_coeffs = List( Float )

    dims = List( Int )

    link_dims = List( Int )

    value = Float

    ##### TODO - adapt the definition
    time_function = Callable

    def _time_function_default( self ):
        return lambda t: t

    def is_essential( self ):
        return self.var == 'u'

    def is_linked( self ):
        return self.link_dofs != []

    def is_constrained( self ):
        '''
        Return true if a DOF is either explicitly prescribed or it depends on other DOFS.
        '''
        return self.is_essential() or self.is_linked()

    def is_natural( self ):
        return self.var == 'f'

    def get_dofs( self ):
        return list( self.dof_numbers.flatten() )

    def setup( self, sctx ):
        '''
        Locate the spatial context.
        '''
        dof_numbers, self.dof_X = self.get_dof_method()
        self.dof_numbers = dof_numbers[:, tuple( self.dims )]

        if self.get_link_dof_method:
            link_dofs, linked_dof_X = self.get_link_dof_method()
            #TODO: test if the shape is the same as dof_numbers
            if len( self.link_dims ) == 0:
                self.link_dims = self.dims
            else:
                if len( self.dims ) != len( self.link_dims ):
                    raise IndexError, 'incompatible dim specification (%d != %d' \
                            % ( len( self.dims ), len( self.link_dims ) )

            link_dofs_arr = link_dofs[:, tuple( self.link_dims )]

            # slice the dof_numbers for the proper direction
            #
            self.bcdof_list = [BCDof( var = self.var, dof = dof,
                                     value = self.value,
                                     link_dofs = [ldof],
                                     link_coeffs = self.link_coeffs,
                                     time_function = self.time_function )
                               for dof, ldof in  zip( self.dof_numbers.flatten(), link_dofs_arr.flatten() ) ]
        else:
            self.bcdof_list = [BCDof( var = self.var, dof = dof,
                                 value = self.value,
                                 link_dofs = self.link_dofs,
                                 link_coeffs = self.link_coeffs,
                                 time_function = self.time_function )
                           for dof in  self.dof_numbers.flatten() ]

        return

    def apply_essential( self, K ):

        for bcond in self.bcdof_list:
            bcond.apply_essential( K )

    def apply( self, step_flag, sctx, K, R, t_n, t_n1 ):

        for bcond in self.bcdof_list:
            bcond.apply( step_flag, sctx, K, R, t_n, t_n1 )

    # register the pipelines for plotting labels and geometry
    #
    mvp_dofs = Trait( MVPointLabels )
    def _mvp_dofs_default( self ):
        return MVPointLabels( name = 'Boundary condition',
                              points = self._get_mvpoints,
                              vectors = self._get_labels,
                              color = ( 0.0, 0.0, 0.882353 ) )

    def _get_mvpoints( self ):
        ## blow up
        return array( self.dof_X, dtype = 'float_' )

    def _get_labels( self ):
        ## blow up
        n_points = self.dof_numbers.shape[0]
        dofs = repeat( -1., n_points * 3 ).reshape( n_points, 3 )
        dofs[:, tuple( self.dims ) ] = self.dof_numbers
        print 'BC - DOFS', dofs
        return dofs

    redraw_button = Button( 'Redraw' )
    def _redraw_button_fired( self ):
        self.mvp_dofs.redraw( label_mode = 'label_vectors' )

    traits_view = View( HSplit( Group( 'var',
                                       'dims',
                                       'value',
                                       'redraw_button' ),
                                Item( 'bcdof_list',
                                     style = 'custom',
                                     editor = bcond_list_editor,
                                     show_label = False ) ),
                        resizable = True,
                        )
