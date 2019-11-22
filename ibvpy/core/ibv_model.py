
import os

from traits.api import \
    Instance, DelegatesTo, Directory, \
    on_trait_change
from traits.util.home_directory import \
    get_home_directory

from .ibv_resource import IBVResource
from .tloop import TLoop


class IBVModel(IBVResource):

    '''Base class for construction of initial boundary value formulation.

    This class should be subclassed to construct for particular applications
    of the time stepping framework. It contains an instance of time loop.
    '''

    def __init__(self, *args, **kw):
        super(IBVModel, self).__init__(*args, **kw)

        # get the directory path of root
        self.dir = self._get_default_dir()
        self.tloop.dir = self.dir

    # service specifiers - used to link the service to this object
    service_class = 'ibvpy.plugins.ibv_model_service.IBVModelService'
    service_attrib = 'ibv_model'

    tloop = Instance(TLoop)

    def _tloop_default(self):
        ''' Default constructor'''
        return TLoop()

    tstepper = DelegatesTo('tloop')
    rtrace_mngr = DelegatesTo('tloop')

    dir = Directory

    def _get_default_dir(self):
        # directory management
        home_dir = get_home_directory()
        mod_base_name = self.__class__.__name__

        sim_data_dir = os.path.join(home_dir, 'simdb', 'simdata')
        if not os.path.exists(sim_data_dir):
            os.mkdir(sim_data_dir)
            print("simdata directory created")

        mod_path = os.path.join(sim_data_dir, mod_base_name)

        if not os.path.exists(mod_path):
            os.mkdir(mod_path)
            print(mod_base_name, " directory created")

        return mod_path

    @on_trait_change('dir')
    def register_ibv_model_dir(self):
        self.tloop.dir = self.dir

    def register_mv_pipelines(self, e):
        '''Delegate the registration to the components.
        '''
        self.tloop.register_mv_pipelines(e)

    # model constructors that can be subclassed
    # @TODO clarify the usage of these methods on examples.
    # How about the state consistency. Should tloop have a backward
    # loop to the ibvmodel?
    def _create_sdomain(self):
        pass

    def _create_trange(self):
        pass

    def _create_bcond(self):
        pass

    def _create_rtrace(self):
        pass


if __name__ == '__main__':
    ibvmodel = IBVModel()
    ibvmodel.configure_traits()
