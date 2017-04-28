

from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from ibvpy.core.rtrace_mngr import RTraceMngr


class RTraceService(HasTraits):

    # Set by envisage when this is offered as a service offer.
    window = Instance('envisage.ui.workbench.workbench_window.WorkbenchWindow')

    rtrace_mngr = Instance(RTraceMngr)

    def _rtrace_mngr_default(self):
        return RTraceMngr()

    traits_view = View(Item('rtrace_mngr@', show_label=False),
                       resizable=True)

if __name__ == '__main__':
    rtrace_mngr_service = RTraceService()
    rtrace_mngr_service.configure_traits()
