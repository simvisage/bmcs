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
from ibvpy.mats.mats1D5.vmats1D5_e import \
    MATS1D5Elastic
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DStrainField, Viz3DTensorField
import numpy as np
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_fe_grid_axisym import XDomainFEGridAxiSym
from simulator.xdomain.xdomain_interface import XDomainFEInterface

from .mlab_decorators import decorate_figure


n_x_e = 20
n_y_e = 5
L_x = 300.0
R_in = 5.0
R_out = 75.0
xd1 = XDomainFEGridAxiSym(coord_min=(0, 0),
                          coord_max=(L_x, R_in),
                          shape=(n_x_e, 2),
                          fets=FETS2D4Q())
xd2 = XDomainFEGridAxiSym(coord_min=(0, R_in),
                          coord_max=(L_x, R_out),
                          shape=(n_x_e, n_y_e),
                          integ_factor=2 * np.pi,
                          fets=FETS2D4Q())
m1 = MATS3DDesmorat(E_1=210000, tau_bar=200.0)
m2 = MATS3DDesmorat()

xd12 = XDomainFEInterface(
    I=xd1.mesh.I[:, -1],
    J=xd2.mesh.I[:, 0],
    fets=FETS1D52ULRHFatigue()
)

left_y = BCSlice(slice=xd1.mesh[0, 0, 0, 0],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xd1.mesh[0, :, 0, :],
                 var='u', dims=[0], value=-0)
right_x = BCSlice(slice=xd1.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=0.1)
bc1 = [left_y, left_x, right_x]

s = Simulator(
    domains=[(xd1, m1),
             (xd2, m2),
             (xd12, MATS1D5Elastic(E_s=10000, E_n=1)),
             ],
    bc=bc1,  # + bc2,
    record={
        'strain': Vis3DStrainField(var='eps_ab'),
        'damage': Vis3DStateField(var='omega_a'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)
s.tloop.k_max = 1000
s.tline.step = 0.01
s.tstep.fe_domain.serialized_subdomains

xd12.hidden = True
s.run()
time.sleep(3)

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

decorate_figure(f_damage, damage_viz)
decorate_figure(f_strain, strain_viz)
mlab.show()
