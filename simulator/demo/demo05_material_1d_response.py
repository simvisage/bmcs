
from bmcs.time_functions import \
    LoadingScenario
from ibvpy.bcond import BCDof
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from simulator.api import Simulator, XDomainSinglePoint

s = Simulator(
    domains=[(XDomainSinglePoint(), MATS3DDesmorat())]
)
bc = BCDof(
    var='f', dof=0, value=-0.001,
    time_function=LoadingScenario()
)
s.bc = [bc]
s.tstep.debug = False
s.tloop.verbose = True
s.run()
s.join_thread()

#print(s.hist.F_t)
#print(s.hist.U_t)

