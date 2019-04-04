'''
Created on 25.03.2019

@author: fseemab
'''
import time

from bmcs.mats.fets1d52ulrhfatigue import FETS1D52ULRHFatigue

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
from mayavi import mlab
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_fe_grid_axisym import XDomainFEGridAxiSym
from simulator.xdomain.xdomain_interface import XDomainFEInterface
from view.window import BMCSWindow

import numpy as np

from .mlab_decorators import decorate_figure
from .viz2d_fw import Viz2DFW, Vis2DFW


ds = 14
n_x_e = 40
n_y_concrete = 5
n_y_steel = 2
L_x = 3 * ds
R_steel = ds / 2
R_concrete = 7 * ds
xd_steel_1 = XDomainFEGridAxiSym(coord_min=(0, 0),
                                 coord_max=(L_x, R_steel),
                                 shape=(n_x_e, n_y_steel),
                                 integ_factor=2 * np.pi,
                                 fets=FETS2D4Q())
xd_concrete_2 = XDomainFEGridAxiSym(coord_min=(0, R_steel),
                                    coord_max=(L_x, R_concrete),
                                    shape=(n_x_e, n_y_concrete),
                                    fets=FETS2D4Q())

m1 = MATS3DDesmorat(E_1=210000, nu=0.3, tau_bar=2000.0)
m2 = MATS3DDesmorat(tau_bar=2.0)


xd12 = XDomainFEInterface(
    I=xd_steel_1.mesh.I[:, -1],
    J=xd_concrete_2.mesh.I[:, 0],
    fets=FETS1D52ULRHFatigue()
)

u_max = 0.3 * 3
right_x_s = BCSlice(slice=xd_steel_1.mesh[-1, :, -1, :],
                    var='u', dims=[0], value=u_max)
right_x_c = BCSlice(slice=xd_concrete_2.mesh[-1, :, -1, :],
                    var='u', dims=[0], value=0)

print(right_x_s.dofs)
bc1 = [right_x_c, right_x_s]

m_interface = MATS1D5DPCumPress()

s = Simulator(
    domains=[(xd_steel_1, m1),
             (xd_concrete_2, m2),
             (xd12, m_interface),
             ],
    bc=bc1,  # + bc2,
    record={
        'Pw': Vis2DFW(bc=right_x_s),
        'strain': Vis3DStrainField(var='eps_ab'),
        'stress': Vis3DStressField(var='sig_ab'),
        'damage': Vis3DStateField(var='omega_a'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)

fw = Viz2DFW(name='Pw', vis2d=s.hist['Pw'])
s.run()
time.sleep(4)
w = BMCSWindow(model=s)
w.viz_sheet.viz2d_list.append(fw)

s.tloop.k_max = 1000
s.tline.step = 0.01
s.tstep.fe_domain.serialized_subdomains

xd12.hidden = True
# s.run()
# time.sleep(3)

w.run()
w.offline = True
w.configure_traits()

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
