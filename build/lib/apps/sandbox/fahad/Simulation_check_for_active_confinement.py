''' 
Created on 06.05.2019 
 
@author: fseemab 
'''
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
from simulator.demo.viz2d_fw import Viz2DFW, Vis2DFW
from simulator.xdomain.xdomain_fe_grid_axisym import XDomainFEGridAxiSym
from simulator.xdomain.xdomain_interface import XDomainFEInterface
from view.window import BMCSWindow

import numpy as np


#from .mlab_decorators import decorate_figure
u_max = 1.5
u_confinement = 0.2
dx = 1
#ds = 12
ds = 1 / np.pi
r_steel = ds / 2.0
r_concrete = ds * 3
xd_steel = XDomainFEGridAxiSym(coord_min=(0, 0),
                               coord_max=(dx, r_steel),
                               shape=(1, 1),
                               integ_factor=2 * np.pi,
                               fets=FETS2D4Q())

xd_concrete = XDomainFEGridAxiSym(coord_min=(0, r_steel),
                                  coord_max=(dx, r_concrete),
                                  shape=(1, 1),
                                  integ_factor=2 * np.pi,
                                  fets=FETS2D4Q())

m_steel = MATS3DElastic(E=200000, nu=0.2)
m_concrete = MATS3DElastic(E=30000, nu=0.3)

xd12 = XDomainFEInterface(
    I=xd_steel.mesh.I[:, -1],
    J=xd_concrete.mesh.I[:, 0],
    fets=FETS1D52ULRH(),
    integ_factor=np.pi * ds
)

right_x_s = BCSlice(slice=xd_steel.mesh[-1, :, -1, :],
                    var='u', dims=[0], value=u_max)
left_x_c = BCSlice(slice=xd_concrete.mesh[0, :, 0, :],
                   var='u', dims=[0], value=0)
top_y_c = BCSlice(slice=xd_concrete.mesh[:, -1, :, -1],
                  var='u', dims=[1], value=-u_confinement)
# bottom_y_c = BCSlice(slice=xd_concrete.mesh[:, 0, :, 0],
# var='u', dims=[1], value=0)
left_x_s = BCSlice(slice=xd_steel.mesh[0, :, 0, :],
                   var='f', dims=[0], value=0)
right_x_c = BCSlice(slice=xd_concrete.mesh[-1, :, -1, :],
                    var='u', dims=[0], value=0)
bc1 = [right_x_s,
       top_y_c, right_x_c, left_x_c]  # bottom_y_c,

#tau_bar = 2.0
E_T = 1000
#s_0 = tau_bar / E_T

m_interface = MATS1D5DPCumPress(  # E_T=E_T, E_N=1000000,
    # gamma=0.0,
    # K=0.0,
    # tau_bar=1.0,
    algorithmic=True)  # omega_fn_type='li',

s = Simulator(
    domains=[(xd_steel, m_steel),
             (xd_concrete, m_concrete),
             (xd12, m_interface),
             ],
    bc=bc1,  # + bc2,
    record={
        'Pw': Vis2DFW(bc_right=right_x_s, bc_left=left_x_s),
        'slip': Vis2DField(var='slip'),
        'strain': Vis3DTensorField(var='eps_ab'),
        'stress': Vis3DTensorField(var='sig_ab'),
        #        'damage': Vis3DStateField(var='omega_a'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)

xd12.hidden = True
s.tloop.k_max = 1000
s.tloop.verbose = True
s.tline.step = 0.005
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
# time.sleep(1)
# decorate_figure(f_strain, strain_viz)
# decorate_figure(f_stress, stress_viz)
# decorate_figure(f_damage, damage_viz)
# mlab.show()
w.configure_traits()
