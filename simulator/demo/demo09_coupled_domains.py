'''This example couples two domains via 
an zero-thickness interface. 

Todo - simplify the domain-staet-xdomain-mesh hierarchy

The dependencies - Simulator - who sets the type of the time stepping
loop and the type of the time step.

Test two independent domains.
'''

import time

from mayavi import mlab

from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_state_field import \
    Vis3DStateField, Viz3DStateField
from ibvpy.mats.viz3d_strain_field import \
    Vis3DStrainField, Viz3DStrainField
import numpy as np
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_fe_grid import XDomainFEGrid
from simulator.xdomain.xdomain_interface import XDomainFEInterface

xdomain1 = XDomainFEGrid(coord_max=(100, 50),
                         shape=(2, 1),
                         fets=FETS2D4Q())
xdomain2 = XDomainFEGrid(coord_min=(0, 60),
                         coord_max=(100, 100),
                         shape=(2, 1),
                         fets=FETS2D4Q())
xdomain12 = XDomainFEInterface(
    I=xdomain1.mesh.I[:, -1],
    J=xdomain2.mesh.I[:, 0]
)

left_y = BCSlice(slice=xdomain1.mesh[0, 0, 0, 0],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xdomain1.mesh[0, :, 0, :],
                 var='u', dims=[0], value=-0.4)
right_x = BCSlice(slice=xdomain1.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=0.0)
bc1 = [left_y, left_x, right_x]
left_y = BCSlice(slice=xdomain2.mesh[0, 0, 0, 0],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xdomain2.mesh[0, :, 0, :],
                 var='u', dims=[0], value=-0.4)
right_x = BCSlice(slice=xdomain2.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=0.0)
bc2 = [left_y, left_x, right_x]

s = Simulator(
    domains=[(xdomain1, MATS3DDesmorat()),
             (xdomain2, MATS3DDesmorat()),
             ],
    bc=bc1 + bc2,
    record={
        #        'strain': Vis3DStrainField(var='eps_ab'),
        'damage': Vis3DStateField(var='omega_a'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)
s.tloop.k_max = 1000
s.tline.step = 0.0
# s.run()
# time.sleep(3)
# print(s.tstep.fe_domain)
print(s.tstep.fe_domain.serialized_subdomains)

# REMARK - The domains have been serialized within
# the fe_domain of the simulator - this is done upon
# the assignment to the tstep that governs the calculation.
# Actually this kind of access is cryptical and should be avoided.
# The problem might be circumvented by providing
# navigation through the domain from the toplevel object -i.e. fe_domain
#
# xd['upper'][ :, -1, :, -1].dofs
# xd['lower'][ :, -1, :, -1].dofs
#
# XD('upper', upper, concrete1,
#    'lower', lower, concrete2,
#    'interface, intf, press_sens)
#
# xdomain.mesh.I[:,-1]
# xdomain.mesh.E[]
#
print(xdomain12.o_pEia)
print(xdomain12.X_pEia)

# strain_viz = Viz3DStrainField(vis3d=s.hist['strain'])
# strain_viz.setup()
# strain_viz.plot(0.0)

# damage_viz = Viz3DStateField(vis3d=s.hist['damage'])
# damage_viz.setup()
# damage_viz.plot(0.0)
# mlab.show()
