
from traits.api import \
    Instance, Property, cached_property, Array

from bmcs.simulator.tstep_state import TStepState
from ibvpy.core.bcond_mngr import BCondMngr
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly


class TStepBC(TStepState):

    # Boundary condition manager
    #
    bcond_mngr = Instance(BCondMngr)

    def _bcond_mngr_default(self):
        return BCondMngr()

    K = Property(
        Instance(SysMtxAssembly),
        depends_on='model_structure_changed'
    )
    '''System matrix with registered essencial boundary conditions.
    '''
    @cached_property
    def _get_K(self):
        K = SysMtxAssembly()
        self.bcond_mngr.apply_essential(K)
        return K
