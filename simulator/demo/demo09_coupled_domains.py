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
from ibvpy.api import MATSEval
from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2DScalarDamage
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz3d_state_field import \
    Vis3DStateField, Viz3DStateField
from ibvpy.mats.viz3d_strain_field import \
    Vis3DStrainField, Viz3DStrainField
from mathkit.matrix_la.dense_mtx import DenseMtx
import numpy as np
from simulator.api import \
    Simulator
from simulator.i_model import IModel
from simulator.xdomain.xdomain_fe_grid import XDomainFEGrid
from simulator.xdomain.xdomain_interface import XDomainFEInterface
import traits.api as tr


@tr.provides(IModel)
class MATS1D5Elastic(MATSEval):

    node_name = "multilinear bond law"

    state_arr_shape = tr.Tuple((0,))

    E_s = tr.Float(100.0, tooltip='Shear stiffness of the interface [MPa]',
                   MAT=True, unit='MPa', symbol='E_\mathrm{s}',
                   desc='Shear-modulus of the interface',
                   auto_set=True, enter_set=True)

    E_n = tr.Float(100.0, tooltip='Normal stiffness of the interface [MPa]',
                   MAT=True, unit='MPa', symbol='E_\mathrm{n}',
                   desc='Normal stiffness of the interface',
                   auto_set=False, enter_set=True)

    state_var_shapes = {}

    D_rs = tr.Property(depends_on='E_n,E_s')

    @tr.cached_property
    def _get_D_rs(self):
        return np.array([[self.E_s, 0],
                         [0, self.E_n]], dtype=np.float_)

    def get_corr_pred(self, u_r, tn1):
        tau = np.einsum(
            'rs,...s->...r',
            self.D_rs, u_r)
        grid_shape = tuple([1 for _ in range(len(u_r.shape[:-1]))])
        D = self.D_rs.reshape(grid_shape + (2, 2))
        return tau, D


n_x_e = 40
xdomain1 = XDomainFEGrid(coord_max=(100, 10),
                         shape=(n_x_e, 3),
                         fets=FETS2D4Q())
xdomain2 = XDomainFEGrid(coord_min=(0, 10),
                         coord_max=(100, 20),
                         shape=(n_x_e, 3),
                         fets=FETS2D4Q())
xdomain12 = XDomainFEInterface(
    I=xdomain1.mesh.I[:, -1],
    J=xdomain2.mesh.I[:, 0],
    fets=FETS1D52ULRHFatigue()
)

left_y = BCSlice(slice=xdomain1.mesh[0, 0, 0, 0],
                 var='u', dims=[1], value=0)
left_x = BCSlice(slice=xdomain1.mesh[0, :, 0, :],
                 var='u', dims=[0], value=-0)
right_x = BCSlice(slice=xdomain1.mesh[-1, :, -1, :],
                  var='u', dims=[0], value=0.1)
bc1 = [left_y, left_x, right_x]

s = Simulator(
    domains=[(xdomain1, MATS2DScalarDamage(algorithmic=False)),
             (xdomain2, MATS2DScalarDamage(algorithmic=False)),
             #             (xdomain3, MATS3DDesmorat()),
             (xdomain12, MATS1D5Elastic(E_s=100, E_n=1)),
             #             (xdomain13, MATS1D5Elastic()),
             ],
    bc=bc1,  # + bc2,
    record={
        'strain': Vis3DStrainField(var='eps_ab'),
        #        'damage': Vis3DStateField(var='omega'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }
)
s.tstep.fe_domain[0].tmodel.omega_fn.f_t = 100.0
s.tloop.k_max = 1000
s.tline.step = 0.05
s.tstep.fe_domain.serialized_subdomains
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


xdomain12.hidden = True
s.run()
time.sleep(4)
strain_viz = Viz3DStrainField(vis3d=s.hist['strain'])
strain_viz.setup()
strain_viz.plot(0.0)

# damage_viz = Viz3DStateField(vis3d=s.hist['damage'])
# damage_viz.setup()
# damage_viz.plot(0.0)
mlab.show()
