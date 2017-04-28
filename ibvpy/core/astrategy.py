
from traits.api import HasTraits, WeakRef, Property

class AStrategyBase( HasTraits ):
    '''
    Interface for adaptive computations.
    
    There are four positions in the algorithm where adaptive changes to the
    spatial and temporal discretization may be applied:
    (1) before the computation starts
    (2) in every iteration (ihandler)
    (3) on failure of the iteration loop (fhandler)
    (4) on equilibrium (ehandler)
    A total of 9 methods is implemented to allow for an flexible adjustment
    of the global algorithm. The default implementation is neutral regarding
    the behavior of the global algorithm. 
    '''

    tloop = WeakRef()

    tstepper = Property()
    def _get_tstepper( self ):
        return self.tloop.tstepper

    # (1) INITIALIZE
    #  
    def initialize_strategy( self ):
        '''Called before the computation starts.'''
        pass

    # (2) LOAD STEP
    #
    def begin_time_step( self, t ):
        '''Prepare a new load step
        '''
        pass

    def end_time_step( self, t ):
        '''Prepare a new load step
        '''
        pass

    # (2) ITERATION (ihandler)
    #  
    def ihandler_needed( self ):
        '''Decides whether the iteration must be adapted.
        True means that ihandler_invoke is called and the time step is restarted.
        False deaginctivates any further ihandler actions.
        '''
        return False

    def ihandler_invoke( self ):
        '''Handle the iteration.'''
        pass

    def ihandler_get_scale( self ):
        '''Scale time step with a factor.'''
        return 1.0

    # (3) EQUILIBRIUM (ehandler)
    #  
    def ehandler_needed( self ):
        '''Decide whether to invoke the equilibrium handler.'''
        return False

    def ehandler_accept( self ):
        '''Decide whether to accept or redo the current time step.'''
        return True

    def ehandler_invoke( self ):
        '''
        Extend algorithm when equilibrium is found
        (but before potentially accepting the state).
        '''
        pass

    def ehandler_get_scale( self ):
        '''Scale time step with a factor.'''
        return 1.0

    # (4) FAILURE (fhandler)
    #  
    def fhandler_get_scale( self ):
        '''
        Handle failure of iteration loop.
        Default: Half the time step.
        '''
        return 1 / 2.
