import time

from mayavi import mlab

from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2DScalarDamage
from ibvpy.mats.viz3d_state_field import \
    Vis3DStateField, Viz3DStateField
from simulator.api import \
    Simulator, XDomainFEGrid


L = 600.0
H = 100.0
L_c = 5.0
a = 5.0
w_max = 0.4
dgrid1 = XDomainFEGrid(coord_max=(L, H),
                       shape=(20, 20),
                       integ_factor=50,
                       fets=FETS2D4Q())
x_x, x_y = dgrid1.mesh.geo_grid.point_x_grid
L_1 = x_x[1, 0]
d_L = L_c - L_1
x_x[1:, :] += d_L * (L - x_x[1:, :]) / (L - L_1)
a_L = a / H
n_a = int(a_L * dgrid1.shape[1])
fixed_right_bc = BCSlice(slice=dgrid1.mesh[-1, 0, -1, 0],
                         var='u', dims=[1], value=0)
fixed_x = BCSlice(slice=dgrid1.mesh[0, n_a:, 0, -1],
                  var='u', dims=[0], value=0)
control_bc = BCSlice(slice=dgrid1.mesh[0, -1, :, -1],
                     var='u', dims=[1], value=-w_max)
s = Simulator(
    model=MATS2DScalarDamage(algorithmic=True),
    xdomain=dgrid1,
    bc=[fixed_right_bc, fixed_x, control_bc],
    record={
        'damage': Vis3DStateField(var='omega'),
    }
)
s.tloop.k_max = 200
s.tline.step = 0.05
s.run()
time.sleep(3)
damage_viz = Viz3DStateField(vis3d=s.hist['damage'])
damage_viz.setup()
damage_viz.plot(0.0)
mlab.show()
