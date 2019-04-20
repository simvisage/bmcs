import time

from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DStrainField, Viz3DTensorField
from mayavi import mlab
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_fe_grid_axisym import XDomainFEGridAxiSym

import numpy as np

from .mlab_decorators import decorate_figure


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
s.run_thread()
time.sleep(5)

mlab.options.backend = 'envisage'

f_strain = mlab.figure()
scene = mlab.get_engine().scenes[-1]
scene.name = 'strain'
strain_viz = Viz3DTensorField(vis3d=s.hist['strain'])
strain_viz.setup()

f_damage = mlab.figure()
scene = mlab.get_engine().scenes[-1]
scene.name = 'damage'
damage_viz = Viz3DScalarField(vis3d=s.hist['damage'])
damage_viz.setup()
damage_viz.use_default_range = True
damage_viz.warp_vector.filter.scale_factor = 100.0
damage_viz.plot(s.tstep.t_n)
damage_viz.plot(0.0)


decorate_figure(f_strain, strain_viz, 200, [70, 20, 0])
decorate_figure(f_damage, damage_viz, 200, [70, 20, 0])

mlab.show()
