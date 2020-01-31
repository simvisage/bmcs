

from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from ibvpy.core.sdomain import SDomain


class SDomainService(HasTraits):

    # Set by envisage when this is offered as a service offer.
    window = Instance('envisage.ui.workbench.workbench_window.WorkbenchWindow')

    sdomain = Instance(SDomain)

    def _sdomain_default(self):
        return SDomain()

    traits_view = View(Item('sdomain@', show_label=False),
                       resizable=True)

if __name__ == '__main__':
    sdomain_service = SDomainService()
    sdomain_service.configure_traits()
