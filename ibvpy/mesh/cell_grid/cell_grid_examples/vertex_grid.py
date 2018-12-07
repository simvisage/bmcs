#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Jul 19, 2010 by: rch

from ibvpy.mesh.cell_grid.cell_grid import CellGrid
from ibvpy.mesh.cell_grid.cell_spec import CellSpec

# Get the intersected element of a one dimensional grid
cell_grid = CellGrid( grid_cell_spec = CellSpec( node_coords = [[-1, -1],
                                                                [-1, 0],
                                                                [-1, 1],
                                                                [ 1, -1],
                                                                [ 1, 0],
                                                                [ 1, 1]]
                                                                ),
                      shape = ( 1, 1 ), coord_max = [ 1., 1. ] )

print('vertex_idx_grid')
print(cell_grid.vertex_idx_grid)
print('vertex slices')
print(cell_grid.vertex_slices)
print('point_x_grid')
print(cell_grid.point_x_grid)
print('point_x_arr')
print(cell_grid.point_x_arr)
print('point_X_grid')
print(cell_grid.point_X_grid)
print('vertex_X_grid', end=' ')
print(cell_grid.vertex_X_grid)

