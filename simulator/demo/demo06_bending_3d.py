'''Example calculating the 3 point bending test 
with just one halve of the specimen included.


Hierarchy of configurable operators to perform an increment 
and iteration step

1. TimeBC - Time dependent boundary conditions 
2. DomainState - multiple uniform domains with state force and stiffness 
2.1 LocalGlobalMapping 
2.1.1 Kinematics - supplies the B matrix
      Can be parameterized at several leels - propose a structure
2.1.2 Discretization
2.2 TModel

Complying structure for response tracing
extracting data from several domains
'''

import time

from mayavi import mlab

from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS3D8H
from ibvpy.mats.mats2D import \
    MATS2DScalarDamage
from ibvpy.mats.mats3D.mats3D_microplane.vmats3D_mpl_d_eeq import \
    MATS3DMplDamageEEQ
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DStrainField, Vis3DStressField, Viz3DTensorField
from simulator.api import \
    Simulator, XDomainFEGrid

from .mlab_decorators import decorate_figure


L = 600.0
H = 100.0
B = 50.0
L_c = 5.0
a = 5.0
w_max = 1
dgrid1 = XDomainFEGrid(dim_u=3,
                       coord_max=(L, H, B),
                       shape=(20, 5, 1),
                       fets=FETS3D8H())
x_x, x_y, y_z = dgrid1.mesh.geo_grid.point_x_grid
L_1 = x_x[1, 0]
d_L = L_c - L_1
x_x[1:, :] += d_L * (L - x_x[1:, :]) / (L - L_1)
a_H = a / H
n_a = int(a_H * dgrid1.shape[1])
fixed_right_bc = BCSlice(slice=dgrid1.mesh[-1, 0, -1, 0, :, :],
                         var='u', dims=[1], value=0)
fixed_x = BCSlice(slice=dgrid1.mesh[0, n_a:, :, 0, :, :],
                  var='u', dims=[0], value=0)
control_bc = BCSlice(slice=dgrid1.mesh[0, -1, :, 0, -1, :],
                     var='u', dims=[1], value=-w_max)

m = MATS3DDesmorat(tau_bar=6.0,
                   E_1=16000,
                   E_2=19000,
                   S=470e-6,
                   K=1300,
                   gamma=1100)

m_mic = MATS3DMplDamageEEQ(
    E=27000,
    nu=0.2,
    epsilon_0=59.0e-6,
    epsilon_f=250.0e-7,
    c_T=0.01
)

s = Simulator(
    domains=[(dgrid1, m_mic)],
    bc=[fixed_right_bc, fixed_x, control_bc],
    record={
        'strain': Vis3DStrainField(var='eps_ab'),
        'stress': Vis3DStressField(var='sig_ab'),
        #        'damage': Vis3DStateField(var='omega_a'),
    }
)

s.tloop.k_max = 200
s.tline.step = 0.05
s.run()
time.sleep(8)

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


# f_damage = mlab.figure()
# scene = mlab.get_engine().scenes[-1]
# scene.name = 'damage'
#
# damage_viz = Viz3DScalarField(vis3d=s.hist['damage'])
# damage_viz.setup()
# damage_viz.lut_manager.use_default_range = True
# damage_viz.warp_vector.filter.scale_factor = 10.0
# damage_viz.plot(s.tstep.t_n)
# damage_viz.plot(0.0)


decorate_figure(f_stress, stress_viz, 800, [300, 40, 0])
decorate_figure(f_strain, strain_viz, 800, [300, 40, 0])
#decorate_figure(f_damage, damage_viz, 800, [300, 40, 0])

mlab.show()
