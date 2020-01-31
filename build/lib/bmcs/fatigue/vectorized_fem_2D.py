'''
Created on Jan 24, 2018

This script demonstrates the looples implementation
of the finite element code for multiphase continuum.
Example (2D discretization)

@author: rch, abaktheer
'''
import os

from ibvpy.api import \
    TLine, BCSlice
from ibvpy.fets import \
    FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2DElastic, MATS2DMplDamageEEQ, MATS2DScalarDamage
from mayavi import mlab
from mayavi.filters.api import ExtractTensorComponents
from tvtk.api import \
    tvtk

from ibvpy.core.vtloop import TimeLoop
from ibvpy.dots.vdots_grid import DOTSGrid
from mathkit.tensor import DELTA23_ab
import numpy as np
from tvtk.tvtk_classes import tvtk_helper


def mlab_view(dataset):
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
                      figure=dataset.class_name[3:])
    engine = mlab.get_engine()
    scene = engine.scenes[0]
    scene.scene.z_plus_view()
    src = mlab.pipeline.add_dataset(dataset)
    warp_vector = mlab.pipeline.warp_vector(src)
    surf = mlab.pipeline.surface(warp_vector)

    etc = ExtractTensorComponents()
    engine.add_filter(etc, warp_vector)
    surface2 = Surface()
    engine.add_filter(surface2, etc)
    etc.filter.scalar_mode = 'component'

    lut = etc.children[0]
    lut.scalar_lut_manager.show_scalar_bar = True
    lut.scalar_lut_manager.show_legend = True
    lut.scalar_lut_manager.scalar_bar.height = 0.8
    lut.scalar_lut_manager.scalar_bar.width = 0.17
    lut.scalar_lut_manager.scalar_bar.position = np.array([0.82,  0.1])


if __name__ == '__main__':

    mats2d = MATS2DScalarDamage(
        stiffness='algorithmic',
        epsilon_0=0.03,
        epsilon_f=1.9 * 1000
    )

    mats2d = MATS2DElastic(
    )

    mats2d = MATS2DMplDamageEEQ(
        epsilon_0=0.03,
        epsilon_f=1.9 * 1000
    )

    fets2d = FETS2D4Q()
    dots = DOTSGrid(L_x=600, L_y=100, n_x=51, n_y=10,
                    fets=fets2d, mats=mats2d)
    xdots = DOTSGrid(L_x=4, L_y=1, n_x=40, n_y=10,
                     fets=fets2d, mats=mats2d)
    tloop = TimeLoop(tline=TLine(min=0, max=1, step=0.1),
                     ts=dots)
    if False:
        tloop.bc_list = [BCSlice(slice=dots.mesh[0, :, 0, :],
                                 var='u', dims=[0, 1], value=0),
                         BCSlice(slice=dots.mesh[25, -1, :, -1],
                                 var='u', dims=[1], value=-50),
                         BCSlice(slice=dots.mesh[-1, :, -1, :],
                                 var='u', dims=[0, 1], value=0)
                         ]
    if True:
        tloop.bc_list = [BCSlice(slice=dots.mesh[0, 0, 0, 0],
                                 var='u', dims=[1], value=0),
                         BCSlice(slice=dots.mesh[-1, 0, -1, 0],
                                 var='u', dims=[1], value=0),
                         BCSlice(slice=dots.mesh[25, -1, :, -1],
                                 var='u', dims=[1], value=-50),
                         BCSlice(slice=dots.mesh[25, -1, :, -1],
                                 var='u', dims=[0], value=0)
                         ]
    if False:
        tloop.bc_list = [BCSlice(slice=dots.mesh[0, :, 0, :],
                                 var='u', dims=[0], value=0),
                         BCSlice(slice=dots.mesh[0, 0, 0, 0],
                                 var='u', dims=[1], value=0),
                         BCSlice(slice=dots.mesh[-1, :, -1, :],
                                 var='u', dims=[1], value=0.01)
                         ]

    cell_class = tvtk.Quad().cell_type
    n_c = fets2d.n_nodal_dofs
    n_i = 4
    n_E, n_i, n_a = dots.x_Eia.shape
    n_Ei = n_E * n_i
    points = np.einsum('Ia,ab->Ib', dots.x_Eia.reshape(-1, n_c), DELTA23_ab)
    ug = tvtk.UnstructuredGrid(points=points)
    ug.set_cells(cell_class, np.arange(n_E * n_i).reshape(n_E, n_i))
    vectors = np.zeros_like(points)
    ug.point_data.vectors = vectors
    ug.point_data.vectors.name = 'displacement'
    warp_arr = tvtk.DoubleArray(name='displacement')
    warp_arr.from_array(vectors)
    ug.point_data.add_array(warp_arr)
    home = os.path.expanduser('~')
    target_dir = os.path.join(home, 'simdb', 'simdata')
    tloop.set(ug=ug, write_dir=target_dir)
    U = tloop.eval()

    record_dofs = dots.mesh[25, -1, :, -1].dofs[:, :, 1].flatten()
    Fd_int_t = np.array(tloop.F_int_record)
    Ud_t = np.array(tloop.U_record)
    import pylab as p
    F_int_t = -np.sum(Fd_int_t[:, record_dofs], axis=1)
    U_t = -Ud_t[:, record_dofs[0]]
    t_arr = np.array(tloop.t_record, dtype=np.float_)
    p.plot(U_t, F_int_t)
    p.show()
