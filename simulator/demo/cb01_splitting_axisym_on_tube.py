import time

from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats3D.mats3D_elastic import MATS3DElastic
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DTensorField, Viz3DTensorField
from mayavi import mlab
from simulator.api import \
    TStepBC
from simulator.xdomain.xdomain_fe_grid_axisym import XDomainFEGridAxiSym

import numpy as np

from .mlab_decorators import decorate_figure

inner_radius = 1
outer_radius = 20
L = 10
E_c = 28e+3
f_c_t = 5.0
eps_0 = f_c_t / E_c
u_0 = L * eps_0
print('u_9', u_0)

xdomain = XDomainFEGridAxiSym(coord_min=(0, inner_radius),
                              coord_max=(L, outer_radius),
                              shape=(10, 10),
                              integ_factor=2 * np.pi,
                              fets=FETS2D4Q())

print(xdomain.B1_Eimabc.shape)
print(xdomain.B0_Eimabc.shape)

m = MATS3DDesmorat()
m = MATS3DElastic(E=28e3, nu=0.3)

left_y = BCSlice(slice=xdomain.mesh[:, 0, :, 0],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xdomain.mesh[0, :, 0, :],
                 var='u', dims=[0], value=-u_0)
right_x = BCSlice(slice=xdomain.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=0.0)

m = TStepBC(
    domains=[(xdomain, m)],
    bc=[left_x, right_x, left_y],
)

m.hist.vis_record = {
    'strain': Vis3DTensorField(var='eps_ab'),
    'stress': Vis3DTensorField(var='sig_ab'),
    #        'damage': Vis3DStateField(var='omega_a'),
    #        'kinematic hardening': Vis3DStateField(var='z_a')
}

s = m.sim
s.tloop.k_max = 1000
s.tline.step = 0.1
s.tloop.verbose = True
s.run()

F_to = m.hist.F_t
U_to = m.hist.U_t
F_right_t = np.sum(F_to[:, right_x.dofs], axis=-1)
F_left_t = np.sum(F_to[:, left_x.dofs], axis=-1)
U_left_t = np.average(U_to[:, left_x.dofs], axis=-1)
print('left')
print('right')
print('U_left')
print(U_left_t)

mlab.options.backend = 'envisage'

f_strain = mlab.figure()
scene = mlab.get_engine().scenes[-1]
scene.name = 'strain'
strain_viz = Viz3DTensorField(vis3d=m.hist['strain'])
strain_viz.setup()
decorate_figure(f_strain, strain_viz, 200, [70, 20, 0])

f_stress = mlab.figure()
scene = mlab.get_engine().scenes[-1]
scene.name = 'stress'
stress_viz = Viz3DTensorField(vis3d=m.hist['stress'])
stress_viz.setup()
decorate_figure(f_stress, stress_viz, 200, [70, 20, 0])

mlab.show()
