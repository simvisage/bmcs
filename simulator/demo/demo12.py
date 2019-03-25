'''
Created on 25.03.2019

@author: fseemab
'''
import time
from mayavi import mlab

from bmcs.mats.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn, \
    MultilinearDamageFn, \
    FRPDamageFn
from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats1D5.vmats1D5_dp_cum_press import \
    MATS1D5DPCumPress
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DStrainField, Vis3DStressField, Viz3DTensorField
import numpy as np
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_fe_grid_axisym import XDomainFEGridAxiSym
from simulator.xdomain.xdomain_interface import XDomainFEInterface

ds = 14
n_x_e = 40
n_y_e1 = 5
n_y_e2 = 2
L_x = 3 * ds
R_in = ds
R_out = 10 * ds / 2
xd1 = XDomainFEGridAxiSym(coord_min=(0, 0),
                          coord_max=(L_x, R_out),
                          shape=(n_x_e, n_y_e1),
                          integ_factor=2 * np.pi,
                          fets=FETS2D4Q())
xd2 = XDomainFEGridAxiSym(coord_min=(0, R_out),
                          coord_max=(L_x, R_out),
                          shape=(n_x_e, n_y_e2),
                          fets=FETS2D4Q())

m1 = MATS3DDesmorat(tau_bar=2.0)
m2 = MATS3DDesmorat(E_1=210000, nu=0.3, tau_bar=2000.0)

xd12 = XDomainFEInterface(
    I=xd1.mesh.I[:, -1],
    J=xd2.mesh.I[:, 0],
    fets=FETS1D52ULRHFatigue()
)

u_max = 0.6 * 3
left_y_c = BCSlice(slice=xd1.mesh[0, :, 0, :],
                   var='u', dims=[0], value=0)
left_x_s = BCSlice(slice=xd2.mesh[0, :, 0, :],
                   var='u', dims=[0], value=u_max)
bc1 = [left_y_c, left_x_s]

m_interface = MATS1D5DPCumPress(E_N=1000, E_T=2000,
                                tau_bar=10)

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

s.tloop.k_max = 2000
s.tline.step = 0.01
s.tstep.fe_domain.serialized_subdomains

xd12.hidden = True
s.run()
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

from .mlab_decorators import decorate_figure

decorate_figure(f_strain, strain_viz)
decorate_figure(f_stress, stress_viz)
decorate_figure(f_damage, damage_viz)
mlab.show()
