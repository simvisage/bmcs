from bmcs.time_functions import \
    LoadingScenario
from ibvpy.bcond import BCDof
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from simulator import Simulator
from simulator.xdomain import XDomainSinglePoint

from .interaction_scripts import run_rerun_test

s = Simulator(
    model=MATS3DDesmorat(),
    xdomain=XDomainSinglePoint()
)
bc = BCDof(
    var='u', dof=0, value=-0.001,
    time_function=LoadingScenario()
)
s.tstep.bcond_mngr.bcond_list = [bc]
s.run()
s.join()
# run_rerun_test(s)
