'''
Created on Apr 23, 2010

@author: alexander
'''

#-------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------

from traits.trait_base \
    import user_name_for
from traitsui.api import \
    Item


#-------------------------------------------------------------------------
#  'Item' class:
#-------------------------------------------------------------------------
class Item (Item):
    """An element in a Traits-based user interface. 
    The 'get_label' method is overloaded in order to add
    the unit string defined in the trait's metadata as 'unit'
    to the label string.
    """

    def get_label(self, ui):
        """ Gets the label to use for a specified Item.
        """
        # Return 'None' if the Item is a separator or spacer:
        if self.is_spacer():
            return None

        label = self.label
        if label != '':
            return label
        name = self.name
        object_ = eval(self.object_, globals(), ui.context)
        trait = object_.base_trait(name)

        # --------------------------------
        # append metadata 'unit' to the label string if it exists:
        dict_ = trait.__dict__
        if 'unit' in dict_:
            unit_str = ' [' + dict_['unit'] + ']'
        else:
            unit_str = ' '
        label = user_name_for(name) + unit_str
        #---------------------------------

        tlabel = trait.label
        if tlabel is None:
            return label

        if isinstance(tlabel, str):
            if tlabel[0:3] == '...':
                return label + tlabel[3:]
            if tlabel[-3:] == '...':
                return tlabel[:-3] + label
            if self.label != '':
                return self.label
            return tlabel

        return tlabel(object, name, label)
