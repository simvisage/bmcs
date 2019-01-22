'''
Created on Feb 11, 2018

@author: rch
'''

import os

from tvtk.api import tvtk

import numpy as np
import traits.api as tr
from view.plot3d.viz3d import Vis3D, Viz3D


class Vis3DField(Vis3D):

    var = tr.Str('<unnamed>')

    def setup(self):
        self.new_dir()
        xdomain = self.sim.xdomain
        fets = xdomain.fets
        DELTA_x_ab = fets.vtk_expand_operator

        vtk_cell_type = fets.vtk_cell_class().cell_type
        n_c = fets.n_nodal_dofs
        n_E, n_i, _ = xdomain.x_Eia.shape
        points = np.einsum(
            'Ia,ab->Ib',
            xdomain.x_Eia.reshape(-1, n_c), DELTA_x_ab
        )
        U = np.zeros_like(points)
        self.ug = tvtk.UnstructuredGrid(points=points)
        vtk_cells = (np.arange(n_E) * n_i)[:, np.newaxis] + \
            np.array(fets.vtk_cell, dtype=np.int_)[np.newaxis, :]
        self.ug.set_cells(vtk_cell_type, vtk_cells)
        self.update(U, 0)


class Viz3DField(Viz3D):

    def plot(self, vot):
        idx = self.vis3d.sim.hist.get_time_idx(vot)
        self.d.file_list = self.vis3d.file_list
        self.d.timestep = idx + 1
