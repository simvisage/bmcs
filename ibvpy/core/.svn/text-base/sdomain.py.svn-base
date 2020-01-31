
from enthought.traits.api import HasTraits, implements, List, Any, WeakRef
from enthought.traits.ui.api import View, Item
from i_sdomain import ISDomain
from scontext import SContext
from ibv_resource import IBVResource
from i_tstepper_eval import ITStepperEval 

class SDomain( IBVResource ):

    implements(ISDomain)
    
    # service specifiers - used to link the service to this object
    service_class = 'ibvpy.plugins.sdomain_service.SDomainService'
    service_attrib = 'sdomain'
    
    subdomains = List([])
    
    xdomains = List([])

    dots = WeakRef( ITStepperEval )
    
    def new_scontext(self):
        '''
        Setup a new spatial context.
        '''
        sctx = SContext()
        sctx.sdomain = self
        return sctx

    def register_mv_pipelines(self,e):
        ''' Register the visualization pipelines in mayavi engine
            (empty by default)
        '''
        pass
    
    traits_view = View( Item('dots@', show_label = False ), 
                        resizable = True, 
                        scrollable = True )
