'''
Created on 25.03.2019

@author: fseemab
'''
import time

from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.fets.fets1D5 import FETS1D52ULRH
from ibvpy.mats.mats1D5.vmats1D5_d import \
    MATS1D5D
from ibvpy.mats.mats1D5.vmats1D5_dp import \
    MATS1D5DP
from ibvpy.mats.mats1D5.vmats1D5_dp_cum_press import \
    MATS1D5DPCumPress
from ibvpy.mats.mats3D.mats3D_elastic.vmats3D_elastic import \
    MATS3DElastic
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz2d_field import \
    Vis2DField, Viz2DField
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DTensorField, Viz3DTensorField
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_fe_grid_axisym import XDomainFEGridAxiSym
from simulator.xdomain.xdomain_interface import XDomainFEInterface
from view.window import BMCSWindow

import numpy as np

from .viz2d_fw import Viz2DFW, Vis2DFW


#from .mlab_decorators import decorate_figure
ds = 14
n_e_ds = 12
phi = 2.5
L_b = phi * ds
n_b = phi * n_e_ds
L_e = ds / n_e_ds
n_ex = 4
L_e_ex = n_ex * L_e
n_y_concrete = 8
n_y_steel = 1
L_x = 2 * L_e_ex + L_b
n_e = 2 * n_ex + n_b
R_steel = ds / 2
R_concrete = 7 * ds
xd_steel_1 = XDomainFEGridAxiSym(coord_min=(0, 0),
                                 coord_max=(L_x, R_steel),
                                 shape=(n_e, n_y_steel),
                                 integ_factor=2 * np.pi,
                                 fets=FETS2D4Q())
print(np.max(xd_steel_1.x_Eia[..., 0]))
xd_concrete_2 = XDomainFEGridAxiSym(coord_min=(0, R_steel),
                                    coord_max=(L_x, R_concrete),
                                    shape=(n_e, n_y_concrete),
                                    integ_factor=2 * np.pi,
                                    fets=FETS2D4Q())

# m_steel = MATS3DDesmorat(E_1=210000, nu=0.3, tau_bar=2000.0)
m_steel = MATS3DElastic(E=210000, nu=0.2)
# m_concrete = MATS3DDesmorat(tau_bar=2.0)
m_concrete = MATS3DElastic(E=28000, nu=0.3)

xd12 = XDomainFEInterface(
    I=xd_steel_1.mesh.I[n_ex:-n_ex, -1],
    J=xd_concrete_2.mesh.I[n_ex:-n_ex, 0],
    fets=FETS1D52ULRH(),
    integ_factor=np.pi * ds
)

u_max = 0.3
#f_max = 4000.0
right_x_s = BCSlice(slice=xd_steel_1.mesh[-1, :, -1, :],
                    var='u', dims=[0], value=u_max)
right_x_c = BCSlice(slice=xd_concrete_2.mesh[0, 1:, 0, :],
                    var='u', dims=[0], value=0)
left_x_s = BCSlice(slice=xd_steel_1.mesh[0, :, 0, :],
                   var='f', dims=[0], value=0)

bc1 = [right_x_c, right_x_s]

tau_bar = 2.0
E_T = 1000
s_0 = tau_bar / E_T
print('s_0', s_0)
m_interface = MATS1D5DPCumPress(  # E_T=E_T, E_N=1000000, omega_fn_type='jirasek',
    algorithmic=True)
#m_interface.omega_fn.trait_set(s_0=s_0, s_f=1000 * s_0)

s = Simulator(
    domains=[(xd_steel_1, m_steel),
             (xd_concrete_2, m_concrete),
             (xd12, m_interface),
             ],
    bc=bc1,  # + bc2,
    record={
        'Pw': Vis2DFW(bc_right=right_x_s, bc_left=left_x_s),
        'slip': Vis2DField(var='slip'),
        'strain': Vis3DTensorField(var='eps_ab'),
        'stress': Vis3DTensorField(var='sig_ab'),
        #       'damage': Vis3DStateField(var='omega_a'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)

xd12.hidden = True

s.tloop.k_max = 1000
s.tline.step = 0.005
s.tloop.verbose = True
s.tstep.fe_domain.serialized_subdomains
fw = Viz2DFW(name='Pw', vis2d=s.hist['Pw'])
fslip = Viz2DField(name='slip', vis2d=s.hist['slip'])

w = BMCSWindow(sim=s)
w.viz_sheet.viz2d_list.append(fw)
w.viz_sheet.viz2d_list.append(fslip)
#w.offline = True
# w.configure_traits()

# mlab.options.backend = 'envisage'
#
# f_strain = mlab.figure()
# scene = mlab.get_engine().scenes[-1]
# scene.name = 'stress'
strain_viz = Viz3DTensorField(vis3d=s.hist['strain'])
# strain_viz.setup()
#strain_viz.warp_vector.filter.scale_factor = 100.0
# strain_viz.plot(s.tstep.t_n)
w.viz_sheet.add_viz3d(strain_viz)

# f_stress = mlab.figure()
# scene = mlab.get_engine().scenes[-1]
# scene.name = 'stress'
stress_viz = Viz3DTensorField(vis3d=s.hist['stress'])
# stress_viz.setup()
#stress_viz.warp_vector.filter.scale_factor = 100.0
# stress_viz.plot(s.tstep.t_n)
w.viz_sheet.add_viz3d(stress_viz)

# f_damage = mlab.figure()
# scene = mlab.get_engine().scenes[-1]
# scene.name = 'damage'
#damage_viz = Viz3DScalarField(vis3d=s.hist['damage'])
# damage_viz.setup()
#damage_viz.warp_vector.filter.scale_factor = 100.0
# damage_viz.plot(s.tstep.t_n)
# w.viz_sheet.add_viz3d(damage_viz)

w.run()
time.sleep(3)
# decorate_figure(f_strain, strain_viz)
# decorate_figure(f_stress, stress_viz)
# decorate_figure(f_damage, damage_viz)
# mlab.show()
w.configure_traits()
