'''This example couples two domains via 
an zero-thickness interface. 

Todo - simplify the domain-staet-xdomain-mesh hierarchy

The dependencies - Simulator - who sets the type of the time stepping
loop and the type of the time step.

Test two independent domains.
'''

import time

from mayavi import mlab

from bmcs.mats.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats1D5.vmats1D5_e import MATS1D5Elastic
from ibvpy.mats.mats2D import \
    MATS2DScalarDamage
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DStrainField, Viz3DTensorField
import numpy as np
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_fe_grid import XDomainFEGrid
from simulator.xdomain.xdomain_interface import XDomainFEInterface


n_x_e = 20
xdomain1 = XDomainFEGrid(coord_min=(0, 0),
                         coord_max=(100, 10),
                         shape=(n_x_e, 3),
                         fets=FETS2D4Q())
xdomain2 = XDomainFEGrid(coord_min=(0, 10),
                         coord_max=(100, 20),
                         shape=(n_x_e, 3),
                         fets=FETS2D4Q())
xdomain12 = XDomainFEInterface(
    I=xdomain1.mesh.I[:, -1],
    J=xdomain2.mesh.I[:, 0],
    fets=FETS1D52ULRHFatigue()
)

left_y = BCSlice(slice=xdomain1.mesh[0, 0, 0, 0],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xdomain1.mesh[0, :, 0, :],
                 var='u', dims=[0], value=-0)
right_x = BCSlice(slice=xdomain1.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=0.1)
bc1 = [left_y, left_x, right_x]

s = Simulator(
    domains=[(xdomain1, MATS2DScalarDamage(algorithmic=False)),
             (xdomain2, MATS2DScalarDamage(algorithmic=False)),
             #             (xdomain3, MATS3DDesmorat()),
             (xdomain12, MATS1D5Elastic(E_s=100, E_n=1)),
             #             (xdomain13, MATS1D5Elastic()),
             ],
    bc=bc1,  # + bc2,
    record={
        'strain': Vis3DStrainField(var='eps_ab'),
        'damage': Vis3DStateField(var='omega'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)
s.tstep.fe_domain[0].tmodel.omega_fn.f_t = 100.0
s.tloop.k_max = 1000
s.tline.step = 0.05
s.tstep.fe_domain.serialized_subdomains

xdomain12.hidden = True
s.run()
time.sleep(5)

mlab.options.backend = 'envisage'
f_strain = mlab.figure()
scene = mlab.get_engine().scenes[-1]
scene.name = 'strain'
strain_viz = Viz3DTensorField(vis3d=s.hist['strain'])
strain_viz.setup()
strain_viz.warp_vector.filter.scale_factor = 100.0
strain_viz.plot(s.tstep.t_n)

f_damage = mlab.figure()
scene = mlab.get_engine().scenes[-1]
scene.name = 'damage'
damage_viz = Viz3DScalarField(vis3d=s.hist['damage'])
damage_viz.setup()
damage_viz.warp_vector.filter.scale_factor = 100.0
damage_viz.plot(s.tstep.t_n)


def decorate_figure(f, viz):
    mlab.view(0, 0, 140,
              np.array([50., 5.,  0.]), figure=f)
    mlab.orientation_axes(viz.src, figure=f)
    axes = mlab.axes(viz.src, figure=f)
    axes.label_text_property.trait_set(
        font_family='times', italic=False
    )
    axes.title_text_property.font_family = 'times'
    axes.axes.trait_set(
        x_label='x', y_label='y', z_label='z'
    )


decorate_figure(f_damage, damage_viz)
decorate_figure(f_strain, strain_viz)
mlab.show()
