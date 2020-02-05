

from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from promod.core.promod import IProMod

class ProModelService(HasTraits):

    # Set by envisage when this is offered as a service offer.
    window = Instance('etsproxy.pyface.workbench.api.WorkbenchWindow')

    promod = Instance(IProMod)

    def _promod_changed(self):
        '''Update the dependent services'''
        pass

    traits_view = View(Item('promod@', show_label = False),
                        resizable = True)

if __name__ == '__main__':
    promod_service = ProModService()
    promod_service.configure_traits()
