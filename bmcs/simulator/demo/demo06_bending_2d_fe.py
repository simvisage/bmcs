from traits.api import \
    Float, Instance, Int, Property

from bmcs.simulator import \
    Simulator, Model, TLoopImplicit, TStepBC
from bmcs.simulator.xdomain import \
    XDomainFEGrid
from bmcs.time_functions import \
    LoadingScenario
from ibvpy.bcond import BCDof
from ibvpy.dots.vdots_grid import \
    DOTSGrid
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2DElastic, MATS2DMplDamageEEQ, MATS2DScalarDamage, MATS2DMplCSDEEQ
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
import numpy as np
from view.ui import BMCSLeafNode

from .interaction_scripts import run_rerun_test

L = 600 / 2.0
L_c = 5.0
dgrid = XDomainFEGrid(L_x=L, L_y=100,
                      integ_factor=50,
                      n_x=3, n_y=3,
                      fets=FETS2D4Q())
x_x, x_y = dgrid.mesh.geo_grid.point_x_grid
L_1 = x_x[1, 0]
d_L = L_c - L_1
x_x[1:, :] += d_L * (L - x_x[1:, :]) / (L - L_1)

s = Simulator(
    model=MATS2DScalarDamage()
)
bc = BCDof(
    var='u', dof=0, value=-0.001,
    time_function=LoadingScenario()
)
s.tstep.xdomain = dgrid
s.tstep.bcond_mngr.bcond_list = [bc]
# run_rerun_test(s)
s.run()
s.join()
