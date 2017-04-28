

from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from ibvpy.core.ibv_model import IBVModel


class IBVModelService(HasTraits):

    # Set by envisage when this is offered as a service offer.
    window = Instance('envisage.ui.workbench.workbench_window.WorkbenchWindow')

    ibv_model = Instance(IBVModel)

    def _ibv_model_default(self):
        return IBVModel()

    def _ibv_model_changed(self):
        '''Update the dependent services'''
        tloop_service = \
            self.window.get_service('ibvpy.plugins.tloop_service.TLoopService')
        tloop_service.tloop = self.ibv_model.tloop

    traits_view = View(Item('ibv_model@', show_label=False),
                       resizable=True)

if __name__ == '__main__':
    ibv_model_service = IBVModelService()
    ibv_model_service.configure_traits()
