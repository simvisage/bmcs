
from traits.api import \
    Interface, Tuple


class IXDomain(Interface):
    '''Mappings between spatial domain representation 
    and between the linear algebr objects. 
    '''
    U_var_shape = Tuple

    state_var_shape = Tuple

    def map_U_to_field(self, eps_eng): pass

    def map_field_to_F(self, eps_tns): pass

    def map_field_to_K(self, tns4): pass
