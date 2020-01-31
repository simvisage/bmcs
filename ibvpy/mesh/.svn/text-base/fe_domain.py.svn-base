
from enthought.traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, Interface, implements, \
     Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
     on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property, \
     This, self, TraitError, Button, Event

from enthought.traits.ui.api import \
    View, Item, Group

from numpy import array, arange

from ibvpy.core.sdomain import \
    SDomain

from ibvpy.core.scontext import \
    SContext

from ibvpy.dots.dots_list_eval import \
    DOTSListEval

from ibvpy.rtrace.rt_domain_list import \
    RTraceDomainList

class FEDomain( SDomain ):
    '''Test the state dependencies within the hierarchical domain representation.
    '''

    changed_structure = Event

    subdomains = List( domain_changed = True )
    @on_trait_change( 'changed_structure' )
    def _validate_subdomains( self ):
        for domain in self.subdomains:
            domain.validate()

    xdomains = List( domain_changed = True )

    serialized_subdomains = List

    def _append_in_series( self, new_subdomain ):
        '''Link the new subdomain at the end of the series.
        '''
        if self.serialized_subdomains:
            last_subdomain = self.serialized_subdomains[-1]
            last_subdomain.next_domain = new_subdomain
            new_subdomain.previous_domain = last_subdomain
        self.serialized_subdomains.append( new_subdomain )

    nonempty_subdomains = Property( depends_on = 'changed_structure' )
    @cached_property
    def _get_nonempty_subdomains( self ):
        d_list = []
        for d in self.serialized_subdomains:
            if d.n_active_elems > 0:
                d_list.append( d )
        return d_list

    n_dofs = Property
    def _get_n_dofs( self ):
        '''Return the total number of dofs in the domain.
        Use the last subdomain's: dof_offset + n_dofs 
        '''
        last_subdomain = self.serialized_subdomains[-1]
        return last_subdomain.dof_offset + last_subdomain.n_dofs

    dof_offset_arr = Property
    def _get_dof_offset_arr( self ):
        '''
        Return array of the dof offsets 
        from serialized subdomains
        '''
        a = array( [domain.dof_offset
                    for domain in self.serialized_subdomains] )
        return a


    #----------------------------------------------------------------------------
    # Methods for time stepper
    #----------------------------------------------------------------------------
    dots = Property( depends_on = 'changed_structure' )
    @cached_property
    def _get_dots( self ):

        return DOTSListEval( sdomain = self,
                             dots_list = [ subdomain.dots
                                           for subdomain
                                           in self.nonempty_subdomains ] )

    def new_scontext( self ):
        '''
        Setup a new spatial context.
        '''
        sctx = SContext()
        sctx.domain_list = self
        return sctx

    #-----------------------------------------------------------------
    # Response tracer background mesh
    #-----------------------------------------------------------------

    rt_bg_domain = Property( depends_on = 'changed_structure' )
    @cached_property
    def _get_rt_bg_domain( self ):
        return RTraceDomainList( subfields = [ subdomain.rt_bg_domain
                                               for subdomain
                                               in self.nonempty_subdomains ],
                                               sd = self )

    def redraw( self ):
        self.rt_bg_domain.redraw()

    #----------------------------------------------------------------------------
    # Methods for extracting ranges from the domain
    #----------------------------------------------------------------------------
    def get_lset_subdomain( self, lset_function ):
        '''@TODO - implement the subdomain selection method
        '''
        raise NotImplementedError

    def get_boundary( self, side = None ):
        '''@todo: - implement the boundary extraction
        '''
        raise NotImplementedError

    def get_interior( self ):
        '''@todo: - implement the boundary extraction
        '''
        raise NotImplementedError

    def __iter__( self ):
        return iter( self.subdomains )

    traits_view = View( Group( 
                               ),
                        resizable = True,
                        scrollable = True,
                        )


