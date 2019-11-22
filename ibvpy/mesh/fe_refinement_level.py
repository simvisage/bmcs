from traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, Interface, \
     Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
     on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property, \
     This, self, TraitError, Dict

from .i_fe_parent_domain import IFEParentDomain
from .fe_subdomain import FESubDomain
from .fe_domain import FEDomain

class FERefinementLevel( FESubDomain ):

    # specialized label
    _tree_label = Str( 'refinement level' )

    def _set_domain( self, value ):
        'reset the domain of this domain'
        if self.parent != None:
            raise TraitError('child FESubDomain cannot be added to FEDomain')
        super( FERefinementLevel, self )._set_domain( value )
    def _get_domain( self ):
        if self.parent != None:
            return self.parent.domain
        return super( FERefinementLevel, self )._get_domain()

    def validate( self ):
        if self.parent != None:
            raise ValueError('only parentless subdomains can be inserted into domain')

    # children domains: list of the instances of the same class
    children = List( This )

    # parent domain
    _parent = This( domain_changed = True )
    parent = Property( This )
    def _set_parent( self, value ):
        'reset the parent of this domain'
        if self._parent:
            # check to see that the changed parent 
            # is within the same domain
            if value.domain != self._parent.domain:
                raise NotImplementedError('Parent change across domains not implemented')
            # unregister in the old parent
            self._parent.children.remove( self )
        else:
            # append the current subdomain at the end of the subdomain
            # series within the domain
            #pass
            value.domain._append_in_series( self )
        # set the new parent
        self._parent = value
        # register the subdomain in the new parent
        self._parent.children.append( self )
    def _get_parent( self ):
        return self._parent

    #---------------------------------------------------------------------------------------
    # Implement the child interface
    #---------------------------------------------------------------------------------------
    # Element refinement representation
    #
    refinement_dict = Dict( changed_structure = True )

    def refine_elem( self, parent_ix, *refinement_args ):
        '''For the specified parent position let the new element decompose.
        '''
        if parent_ix in self.refinement_dict:
            raise ValueError('element %s already refined' % repr(parent_ix))

        # the element is deactivated in the parent domain
        self.refinement_dict[ parent_ix ] = refinement_args
        self.parent.deactivate( parent_ix )
