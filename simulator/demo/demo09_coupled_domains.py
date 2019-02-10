'''This example couples two domains via 
an zero-thickness interface. 
'''

import time

from mayavi import mlab
from traits.api import provides, List
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
from simulator.i_xdomain import IXDomain
from simulator.xdomain.xdomain_transform import XDomainFEGridTransform
from view.ui.bmcs_tree_node import BMCSTreeNode


@provides(IXDomain)
class XDomain(BMCSTreeNode):
    subdomains = List()

    def __init__(self, *args, **kw):
        super().__init__(**kw)
        self.subdomains = args


xdomain1 = XDomainFEGridTransform(coord_max=(100, 10),
                                  shape=(2, 1),
                                  fets=FETS2D4Q())
xdomain2 = XDomainFEGridTransform(coord_min=(100, 10),
                                  coord_max=(100, 20),
                                  shape=(2, 1),
                                  fets=FETS2D4Q())
xdomain = XDomain([xdomain1, xdomain2])

left_y = BCSlice(slice=xdomain1.mesh[0, 0, 0, 0],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xdomain1.mesh[0, :, 0, :],
                 var='u', dims=[0], value=-0.4)
right_x = BCSlice(slice=xdomain1.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=0.0)

s = Simulator(
    model=MATS3DDesmorat(),
    xdomain=xdomain1,
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

damage_viz = Viz3DStateField(vis3d=s.hist['damage'])
damage_viz.setup()
damage_viz.plot(0.0)
mlab.show()
