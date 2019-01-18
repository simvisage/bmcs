from bmcs.simulator import \
    Simulator
from bmcs.simulator.xdomain import \
    XDomainFEGrid
from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2DScalarDamage

from .interaction_scripts import run_rerun_test

L = 600.0
H = 100.0
L_c = 5.0
a = 50.0
w_max = 0.3
dgrid1 = XDomainFEGrid(L_x=L, L_y=H,
                       integ_factor=50,
                       n_x=5, n_y=2,
                       fets=FETS2D4Q())
x_x, x_y = dgrid1.mesh.geo_grid.point_x_grid
L_1 = x_x[1, 0]
d_L = L_c - L_1
x_x[1:, :] += d_L * (L - x_x[1:, :]) / (L - L_1)
s = Simulator(
    model=MATS2DScalarDamage(),
    xdomain=dgrid1
)
a_L = a / H
n_a = int(a_L * dgrid1.n_y)
fixed_right_bc = BCSlice(slice=dgrid1.mesh[-1, 0, -1, 0],
                         var='u', dims=[1], value=0)
fixed_x = BCSlice(slice=dgrid1.mesh[0, n_a:, 0, -1],
                  var='u', dims=[0], value=0)
control_bc = BCSlice(slice=dgrid1.mesh[0, -1, :, -1],
                     var='u', dims=[1], value=-w_max)
s.tstep.bcond_mngr.bcond_list = [
    fixed_right_bc,
    fixed_x,
    control_bc
]
s.run()
s.join()
print(s.tstep.U_n)
