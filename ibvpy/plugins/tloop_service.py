

from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from ibvpy.core.tloop import TLoop


class TLoopService(HasTraits):

    # Set by envisage when this is offered as a service offer.
    window = Instance('envisage.ui.workbench.workbench_window.WorkbenchWindow')

    tloop = Instance(TLoop)

    def _tloop_default(self):
        return TLoop()

    def _tloop_changed(self):
        '''Update the dependent services'''
        tstepper_service = \
            self.window.get_service(
                'ibvpy.plugins.tstepper_service.TStepperService')
        tstepper_service.tstepper = self.tloop.tstepper

    traits_view = View(Item('tloop@', show_label=False),
                       resizable=True)

if __name__ == '__main__':
    tloop_service = TLoopService()
    tloop_service.configure_traits()
