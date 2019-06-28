'''This example couples two domains via 
an zero-thickness interface. 

Todo - simplify the domain-staet-xdomain-mesh hierarchy

The dependencies - Simulator - who sets the type of the time stepping
loop and the type of the time step.

Test two independent domains.

@todo:
- define the damage function with parameters realistic for concrete
- define the UI components representing the scale of the problem
- sort XDomains as mix-in class - define the class hierarchy and examples 
- reproduce the pullout from the fatigue paper
- test the the pressure sensitive bond
- include the microplane model with parameters calibrated on 3d bending
- instead of mlab get the bmcs window
- reproduce splitting
'''

import time

from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn, \
    MultilinearDamageFn, \
    FRPDamageFn
from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.fets.fets1D5.fets1d52ulrh import FETS1D52ULRH
from ibvpy.mats.mats1D5.vmats1D5_dp import \
    MATS1D5DP
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DStrainField, Vis3DStressField, Viz3DTensorField
from mayavi import mlab
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_fe_grid_axisym import XDomainFEGridAxiSym
from simulator.xdomain.xdomain_interface import XDomainFEInterface

import numpy as np

from .mlab_decorators import decorate_figure


n_x_e = 30
n_y_e = 5
L_x = 300.0
R_in = 5.0
R_out = 75.0 / 2
xd1 = XDomainFEGridAxiSym(coord_min=(0, 0),
                          coord_max=(L_x, R_in),
                          shape=(n_x_e, 2),
                          fets=FETS2D4Q())
xd2 = XDomainFEGridAxiSym(coord_min=(0, R_in),
                          coord_max=(L_x, R_out),
                          shape=(n_x_e, n_y_e),
                          integ_factor=2 * np.pi,
                          fets=FETS2D4Q())
m1 = MATS3DDesmorat(E_1=210000, nu=0.3, tau_bar=2000.0)
m2 = MATS3DDesmorat(tau_bar=2.0)

xd12 = XDomainFEInterface(
    I=xd1.mesh.I[:, -1],
    J=xd2.mesh.I[:, 0],
    fets=FETS1D52ULRH()
)

u_max = 0.6 * 3
left_y = BCSlice(slice=xd1.mesh[0, 0, 0, 0],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xd1.mesh[0, :, 0, :],
                 var='u', dims=[0], value=-u_max)
right_x = BCSlice(slice=xd1.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=u_max)
bc1 = [left_y, left_x, right_x]

m_interface = MATS1D5DP(E_N=1000, E_T=2000,
                        tau_bar=10)

m_interface.omega_fn = FRPDamageFn()
s = Simulator(
    domains=[(xd1, m1),
             (xd2, m2),
             (xd12, m_interface),
             ],
    bc=bc1,  # + bc2,
    record={
        'strain': Vis3DStrainField(var='eps_ab'),
        'stress': Vis3DStressField(var='sig_ab'),
        'damage': Vis3DStateField(var='omega_a'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)

s.tloop.k_max = 1000
s.tline.step = 0.01
s.tstep.fe_domain.serialized_subdomains
s.tloop.verbose = True

xd12.hidden = True
s.run_thread()
time.sleep(3)

mlab.options.backend = 'envisage'

f_strain = mlab.figure()
scene = mlab.get_engine().scenes[-1]
scene.name = 'stress'
strain_viz = Viz3DTensorField(vis3d=s.hist['strain'])
strain_viz.setup()
strain_viz.warp_vector.filter.scale_factor = 100.0
strain_viz.plot(s.tstep.t_n)

f_stress = mlab.figure()
scene = mlab.get_engine().scenes[-1]
scene.name = 'stress'
stress_viz = Viz3DTensorField(vis3d=s.hist['stress'])
stress_viz.setup()
stress_viz.warp_vector.filter.scale_factor = 100.0
stress_viz.plot(s.tstep.t_n)

f_damage = mlab.figure()
scene = mlab.get_engine().scenes[-1]
scene.name = 'damage'
damage_viz = Viz3DScalarField(vis3d=s.hist['damage'])
damage_viz.setup()
damage_viz.warp_vector.filter.scale_factor = 100.0
damage_viz.plot(s.tstep.t_n)


decorate_figure(f_strain, strain_viz)
decorate_figure(f_stress, stress_viz)
decorate_figure(f_damage, damage_viz)
mlab.show()