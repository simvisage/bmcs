import time

from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats2D.mats2D_elastic import MATS2DElastic
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DTensorField, Viz3DTensorField
from mayavi import mlab
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_fe_grid import XDomainFEGrid

import numpy as np

from .mlab_decorators import decorate_figure


thickness = 1

xdomain = XDomainFEGrid(coord_max=(1, 1),
                        shape=(1, 1),
                        integ_factor=thickness,
                        fets=FETS2D4Q())

# print(xdomain.B1_Eimabc.shape)
# print(xdomain.B0_Eimabc.shape)

m = MATS3DDesmorat()
m = MATS2DElastic(E=1, nu=0)

left_y = BCSlice(slice=xdomain.mesh[0, :, 0, :],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xdomain.mesh[0, :, 0, :],
                 var='u', dims=[0], value=1)
right_x = BCSlice(slice=xdomain.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=0.0)

s = Simulator(
    domains=[(xdomain, m)],
    bc=[left_x, right_x, left_y],
    record={
        'strain': Vis3DTensorField(var='eps_ab'),
        #        'damage': Vis3DStateField(var='omega_a'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)
s.tloop.k_max = 1000
s.tline.step = 1
s.tloop.verbose = True
s.run()

print('area', thickness)
F_ti = s.hist.F_t
print('left')
print(np.sum(F_ti[-1, right_x.dofs]))
print('right')
print(np.sum(F_ti[-1, left_x.dofs]))

mlab.options.backend = 'envisage'

f_strain = mlab.figure()
scene = mlab.get_engine().scenes[-1]
scene.name = 'strain'
strain_viz = Viz3DTensorField(vis3d=s.hist['strain'])
strain_viz.setup()

decorate_figure(f_strain, strain_viz, 200, [70, 20, 0])

mlab.show()
