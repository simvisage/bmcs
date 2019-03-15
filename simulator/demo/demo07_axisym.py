import time

from mayavi import mlab
from traits.api import \
    provides, Property, Array, cached_property

from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_state_field import \
    Vis3DStateField, Viz3DStateField
from ibvpy.mats.viz3d_strain_field import \
    Vis3DStrainField, Viz3DStrainField
from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
import numpy as np
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_fe_grid_axisym import XDomainFEGridAxiSym

xdomain = XDomainFEGridAxiSym(coord_max=(150, 50),
                              shape=(15, 5),
                              integ_factor=2 * np.pi,
                              fets=FETS2D4Q())

print(xdomain.B1_Eimabc.shape)
print(xdomain.B0_Eimabc.shape)

m = MATS3DDesmorat()

left_y = BCSlice(slice=xdomain.mesh[0, :, 0, :],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xdomain.mesh[0, :, 0, :],
                 var='u', dims=[0], value=0.5)
right_x = BCSlice(slice=xdomain.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=0.0)

s = Simulator(
    domains=[(xdomain, m)],
    bc=[left_x, right_x, left_y],
    record={
        'strain': Vis3DStrainField(var='eps_ab'),
        'damage': Vis3DStateField(var='omega_a'),
        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)
s.tloop.k_max = 1000
s.tline.step = 0.05
s.run()
time.sleep(5)

strain_viz = Viz3DStrainField(vis3d=s.hist['strain'])
strain_viz.setup()
damage_viz = Viz3DStateField(vis3d=s.hist['damage'])
damage_viz.setup()
damage_viz.plot(0.0)
mlab.show()
