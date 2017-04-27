
from traits.api import \
    HasTraits, Instance, Delegate, HasStrictTraits, Str

class IBVResource(HasTraits):
    '''Base class for components of the boundary value problem.

    This class defines the generic behavior of the IBVResources,
    namely, IBVModel. TLoop, TStepper, RTraceMngr and SDomain.

    It makes the component classes available to the services
    within the application framework IBVPyApp.
    '''

    service_class = Str('')
    service_attrib = Str('')

    def bind_services(self, window):
        '''Issue the binding between the service and resource
        '''
        ibv_service = window.get_service(self.service_class)
        setattr(ibv_service, self.service_attrib, self)

    def register_mv_pipeline(self, e):
        pass
