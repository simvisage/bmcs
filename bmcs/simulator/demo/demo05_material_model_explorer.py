
from traits.api import \
    Instance
from bmcs.simulator import Simulator
from bmcs.time_functions import \
    LoadingScenario
from ibvpy.bcond import BCDof


class SimMATSExplore(Simulator):
    '''
    Simulate the loading histories of a material point in 2D space.
    '''

    bc = Instance(BCDof)

    def _bc_default(self):
        return BCDof(
            var='u', dof=0, value=-0.001,
            time_function=self.loading_scenario
        )

    #=========================================================================
    # Test setup parameters
    #=========================================================================
    loading_scenario = Instance(
        LoadingScenario,
        report=True,
        desc='object defining the loading scenario'
    )

    def _loading_scenario_default(self):
        return LoadingScenario()


if __name__ == '__main__':
    from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
        MATS3DDesmorat

    s = SimMATSExplore(
        model=MATS3DDesmorat()
    )

    s.run()
    s.join()
