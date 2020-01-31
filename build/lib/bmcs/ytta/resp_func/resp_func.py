
from enthought.traits.api import \
    HasTraits, List, Interface, Str, Float
    
#-----------------------------------------------------------------------------------
#                             RESPONSE FUNCTIONS                                  #
#-----------------------------------------------------------------------------------
class IResponseFunction( Interface ):
    """
    Abstract class representing the single response.
    As a realization any function class with the get_value member
    may be included.
    """
    def get_ydata( self, u ):
        '''
        Return the force/stress for a given displacement/strain.
        '''

    def get_values( self ):
        '''
        Return two arrays for stress/strain (force/displacement) response.
        '''
        

class ResponseFunctionBase( HasTraits ):

    param_names = List( Str )
    
    def identify_parameters( self ):
        '''
        Extract the traits that are floating points and can be associated 
        with a statistical distribution.
        '''
        if len( self.param_names ) > 0:
            params = {}
            for param_name in self.param_names:
                params[ param_name ] = self.traits()[param_name]
            print(params)
            return params 
        else:
            params = {}
            for name, trait in list(self.traits().items()):
                if trait.trait_type.__class__ is Float:
                    params[name] = trait
            return params


    def add_listeners( self ):
        self.on_trait_change( self.get_value, 'boundary.+modified,'\
                                        'approach.+modified,'\
                                        'material.+modified,'\
                                        'plot.+modified,'\
                                        'geometry.+modified,'\
                                        'boundary.type.+modified' )
        
    def remove_listeners( self ):
        self.on_trait_change( self.get_value, 'boundary.+modified,'\
                                        'approach.+modified,'\
                                        'material.+modified,'\
                                        'plot.+modified,'\
                                        'geometry.+modified,'\
                                        'boundary.type.+modified',
                                                remove=True )

    def __call__( self, xvalue ):
        '''Return the value of stress/force for a given strain/displacement.
        '''
