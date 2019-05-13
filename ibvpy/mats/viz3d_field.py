'''
Created on Feb 11, 2018

@author: rch
'''

import os

from tvtk.api import tvtk
from view.plot3d.viz3d import Vis3D, Viz3D

import numpy as np
import traits.api as tr


class Vis3DField(Vis3D):

    var = tr.Str('<unnamed>')

    def setup(self):
        self.new_dir()
        # make a loop over the XDomainModel
        fe_domain = self.sim.tstep.fe_domain
        vtk_point_list = []
        vtk_cell_list = []
        vtk_cell_offset_list = []
        vtk_cell_types_list = []
        point_offset = 0
        cell_offset = 0
        for domain in fe_domain:
            xdomain = domain.xdomain
            var_function = domain.tmodel.var_dict.get(self.var, None)
            if var_function == None or xdomain.hidden:
                continue
            fets = xdomain.fets
            DELTA_x_ab = fets.vtk_expand_operator
            n_c = fets.n_nodal_dofs
            vtk_points = np.einsum(
                'Ia,ab->Ib',
                xdomain.x_Eia.reshape(-1, n_c), DELTA_x_ab
            )
            vtk_point_list.append(vtk_points)
            cells, cell_offsets, cell_types = xdomain.get_vtk_cell_data(
                'nodes', point_offset, cell_offset)
            point_offset += vtk_points.shape[0]
            cell_offset += cells.shape[0]
            vtk_cell_list.append(cells)
            vtk_cell_offset_list.append(cell_offsets)
            vtk_cell_types_list.append(cell_types)
        vtk_cell_types = np.hstack(vtk_cell_types_list)
        vtk_cell_offsets = np.hstack(vtk_cell_offset_list)
        vtk_cells = np.hstack(vtk_cell_list)
        n_cells = vtk_cell_types.shape[0]
        vtk_cell_array = tvtk.CellArray()
        vtk_cell_array.set_cells(n_cells, vtk_cells)
        self.ug = tvtk.UnstructuredGrid(points=np.vstack(vtk_point_list))
        self.ug.set_cells(vtk_cell_types,
                          vtk_cell_offsets,
                          vtk_cell_array)


class Viz3DField(Viz3D):

    visible = tr.Property(tr.Bool)

    def _set_visible(self, visible):
        self.d.visible = visible

    def _get_visible(self):
        return self.d.visible

    def plot(self, vot):
        idx = self.vis3d.sim.hist.get_time_idx(vot)
        self.d.file_list = self.vis3d.file_list
        self.d.timestep = idx
