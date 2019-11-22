
from traits.api import provides, List, Any, WeakRef
from traitsui.api import View, Item

from .i_sdomain import ISDomain
from .i_tstepper_eval import ITStepperEval
from .ibv_resource import IBVResource
from .scontext import SContext


@provides(ISDomain)
class SDomain(IBVResource):

    # service specifiers - used to link the service to this object
    service_class = 'ibvpy.plugins.sdomain_service.SDomainService'
    service_attrib = 'sdomain'

    subdomains = List([])

    xdomains = List([])

    dots = WeakRef(ITStepperEval)

    def new_scontext(self):
        '''
        Setup a new spatial context.
        '''
        sctx = SContext()
        sctx.sdomain = self
        return sctx

    def register_mv_pipelines(self, e):
        ''' Register the visualization pipelines in mayavi engine
            (empty by default)
        '''
        pass

    traits_view = View(Item('dots@', show_label=False),
                       resizable=True,
                       scrollable=True)
