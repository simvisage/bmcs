'''This example uses two dimensional grid as xdomain  
with an additional geometric transformation geo_
The transformation automatically expands the strain  
tensor into 3D and augments the lateral strain 
component to zero, rendering planar strain state.

Thus, the material model gets automatically 3D 
while 2D equilibrium is considered at the global level.

The Desmorat model is applied.

Clarify the dependency between the geo_transform
and dimensionality - geo_transform does not necessarily
require the expansion to 3D plane strain.

Next step: provide the coupling of dofs 
'''

import time

from mayavi import mlab

from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DStrainField, Viz3DTensorField
import numpy as np
from simulator.api import \
    Simulator
from simulator.xdomain.xdomain_transform import XDomainFEGridTransform


def geo_trans(points):
    '''Transform a grid geometry to alayered, 
    sinusoidal discretization.
    '''
    x, y = points.T
    return np.c_[x, 50 * np.sin(x / 150 * np.pi) + y]


xdomain = XDomainFEGridTransform(coord_max=(150, 50),
                                 shape=(15, 5),
                                 geo_transform=geo_trans,
                                 fets=FETS2D4Q())

left_y = BCSlice(slice=xdomain.mesh[0, 0, 0, 0],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xdomain.mesh[0, :, 0, :],
                 var='u', dims=[0], value=-0.4)
right_x = BCSlice(slice=xdomain.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=0.0)

s = Simulator(
    domains=[(xdomain, MATS3DDesmorat())],
    bc=[left_x, right_x, left_y],
    record={
        'strain': Vis3DStrainField(var='eps_ab'),
        'damage': Vis3DStateField(var='omega_a'),
        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)
s.tloop.k_max = 1000
s.tline.step = 0.05
s.run()
time.sleep(2)

# strain_viz = Viz3DStrainField(vis3d=s.hist['strain'])
# strain_viz.setup()
# strain_viz.plot(0.0)

damage_viz = Viz3DScalarField(vis3d=s.hist['damage'])
damage_viz.setup()
damage_viz.plot(0.0)
mlab.show()
