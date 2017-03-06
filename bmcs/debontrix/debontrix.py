'''
Created on Dec 20, 2016

This script demonstrates the looples implementation
of the finite element code for multilayer composite.

The current version demonstrating a uni-axial two-layer
continuum discretized with linear shape functions

Planned extensions:
   * Based on the element template generate the shape functions 
     based on the partition of unity concept. Generate symbolic 
     shape functions using sympy
   * Use sympy to construct the shape function derivatives up to
     the required order
   * Introduce the differential operator putting the differential
     terms together to represent the weak form
   * Introduce the integration template within an element template
   * Introduce the visualization template for an element.
   * Single layer works
   * Put the object into a view window.
   * Use a non-linear material model to demonstrate bond.
   * Use a non-linear material model to demonstrate strain-localization
   * Provide visualization of shear flow and slip.
   * Questions to be answered using this app

   * What is the relation to multi-dimensional case?
   * Generalization 
@author: rch
'''

from traits.api import \
    Int, implements, Array, \
    List, Property, cached_property

from ibvpy.api import \
    IFETSEval, FETSEval
from ibvpy.dots.dots_grid_eval import DOTSGridEval
import numpy as np
import sympy as sp


n_C = 2

r_ = sp.symbols('r')


class FETS1D2L(FETSEval):
    '''Example of a finite element definition.
    '''

    implements(IFETSEval)

    dim_slice = slice(0, 1)
    n_e_dofs = Int(2 * n_C)
    n_nodal_dofs = Int(n_C)
    dof_r = Array(value=[[-1], [1]])
    geo_r = Array(value=[[-1], [1]])
    vtk_r = Array(value=[[-1.], [1.]])
    vtk_cells = [[0, 1]]
    vtk_cell_types = 'Line'

    r_m = Array(value=[[-1], [1]], dtype=np.float_)
    w_m = Array(value=[1, 1], dtype=np.float_)

    Nr_i_geo = List([(1 - r_) / 2.0,
                     (1 + r_) / 2.0, ])

    dNr_i_geo = List([- 1.0 / 2.0,
                      1.0 / 2.0, ])

    Nr_i = Nr_i_geo
    dNr_i = dNr_i_geo

    N_mi_geo = Property()

    @cached_property
    def _get_N_mi_geo(self):
        return self.get_N_mi(sp.Matrix(self.Nr_i_geo, dtype=np.float_))

    dN_mid_geo = Property()

    @cached_property
    def _get_dN_mid_geo(self):
        return self.get_dN_mid(sp.Matrix(self.dNr_i_geo, dtype=np.float_))

    N_mi = Property()

    @cached_property
    def _get_N_mi(self):
        return self.get_N_mi(sp.Matrix(self.Nr_i, dtype=np.float_))

    dN_mid = Property()

    @cached_property
    def _get_dN_mid(self):
        return self.get_dN_mid(sp.Matrix(self.dNr_i, dtype=np.float_))

    def get_N_mi(self, Nr_i):
        return np.array([Nr_i.subs(r_, r)
                         for r in self.r_m], dtype=np.float_)

    def get_dN_mid(self, dNr_i):
        dN_mdi = np.array([[dNr_i.subs(r_, r)]
                           for r in self.r_m], dtype=np.float_)
        return np.einsum('mdi->mid', dN_mdi)


if __name__ == '__main__':
    import pylab as p

    dots = DOTSGridEval(fets_eval=FETS1D2L(),
                        n_E=5,
                        L_x=1.0,
                        G=1.0)

    print 's_C', dots.X_J, dots.u_C
    fig = p.figure()
    dots.plot(fig)
    p.show()
