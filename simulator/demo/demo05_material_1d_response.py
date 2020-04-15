
from bmcs.time_functions import \
    LoadingScenario
from ibvpy.bcond import BCDof
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from simulator.api import TStepBC, XDomainSinglePoint

bc = BCDof(
    var='f', dof=0, value=-0.001,
    time_function=LoadingScenario()
)
model = TStepBC(
    domains=[(XDomainSinglePoint(), MATS3DDesmorat())],
    bc=[bc],
    debug=False
)
s = model.sim
s.tloop.verbose = True
s.run_thread()
s.join_thread()

print(model.hist.F_t)
print(model.hist.U_t)
