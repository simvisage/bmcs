# Author: Gael Varoquaux <gael dot varoquaux at normalesup.org>
# Copyright (c) 2008, Enthought, Inc.
# License: BSD style.

from mayavi import mlab
from mayavi.sources.vtk_data_source import VTKDataSource
from numpy import array, random, linspace, pi, ravel, cos, sin, empty
from tvtk.api import tvtk


def unstructured_grid():
    points = array([[0, 1.2, 0.6], [1, 0, 0], [0, 1, 0], [1, 1, 1],  # tetra
                    [1, 0, -0.5], [2, 0, 0], [2, 1.5, 0], [0, 1, 0],
                    [1, 0, 0], [1.5, -0.2, 1], [1.6, 1, 1.5], [1, 1, 1],  # Hex
                    ], 'f')
    # The cells
    cells = array([4, 0, 1, 2, 3,  # tetra
                   8, 4, 5, 6, 7, 8, 9, 10, 11  # hex
                   ])
    # The offsets for the cells, i.e. the indices where the cells
    # start.
    offset = array([0, 5])
    tetra_type = tvtk.Tetra().cell_type  # VTK_TETRA == 10
    hex_type = tvtk.Hexahedron().cell_type  # VTK_HEXAHEDRON == 12
    cell_types = array([tetra_type, hex_type])
    # Create the array of cells unambiguously.
    cell_array = tvtk.CellArray()
    cell_array.set_cells(2, cells)
    # Now create the UG.
    ug = tvtk.UnstructuredGrid(points=points)
    # Now just set the cell types and reuse the ug locations and cells.
    print(cell_types)
    print(offset)
    print(cell_array)
    ug.set_cells(cell_types, offset, cell_array)
    scalars = random.random(points.shape[0])
    ug.point_data.scalars = scalars
    ug.point_data.scalars.name = 'scalars'
    return ug


def view(dataset):
    """ Open up a mayavi scene and display the dataset in it.
    """
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
                      figure=dataset.class_name[3:])
    surf = mlab.pipeline.surface(dataset, opacity=0.1)
    mlab.pipeline.surface(mlab.pipeline.extract_edges(surf),
                          color=(0, 0, 0), )


@mlab.show
def main():
    view(unstructured_grid())


if __name__ == '__main__':
    main()
