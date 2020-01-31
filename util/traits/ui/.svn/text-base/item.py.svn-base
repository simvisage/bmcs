'''
Created on Apr 23, 2010

@author: alexander
'''

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.ui.api import \
    Item

import re

from string \
    import find, rfind

from enthought.traits.api \
    import Instance, Str, Float, Range, Constant, Bool, Callable, Property, \
           Delegate, Undefined, cached_property

from enthought.traits.trait_base \
    import user_name_for


#-------------------------------------------------------------------------------
#  'Item' class: 
#-------------------------------------------------------------------------------

class Item ( Item ):
    """An element in a Traits-based user interface. 
    The 'get_label' method is overloaded in order to add
    the unit string defined in the trait's metadata as 'unit'
    to the label string.
    """
    
    def get_label ( self, ui ):
        """ Gets the label to use for a specified Item.
        """
        # Return 'None' if the Item is a separator or spacer:
        if self.is_spacer():
            return None

        label = self.label
        if label != '':
            return label
        name   = self.name
        object = eval( self.object_, globals(), ui.context )
        trait  = object.base_trait( name )

        # --------------------------------
        # append metadata 'unit' to the label string if it exists: 
        dict = trait.__dict__
        if dict.has_key( 'unit' ):
            unit_str = ' [' + dict['unit'] +']'
        else:
            unit_str = ' '
        label  = user_name_for( name ) + unit_str 
        #---------------------------------

        tlabel = trait.label
        if tlabel is None:
            return label
            
        if isinstance( tlabel, basestring ):
            if tlabel[0:3] == '...':
                return label + tlabel[3:]
            if tlabel[-3:] == '...':
                return tlabel[:-3] + label
            if self.label != '':
                return self.label
            return tlabel
            
        return tlabel( object, name, label )



    