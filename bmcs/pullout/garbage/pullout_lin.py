'''
Created on Dec 20, 2016

This script demonstrates the looples implementation
of the finite element code for multilayer composite.

The current version demonstrating a uni-axial two-layer
continuum discretized with linear shape functions

Planned extensions:

Finite element formulation
==========================
   * Element template should generate the shape functions 
     based on the partition of unity concept. Generate symbolic 
     shape functions using sympy
   * Use sympy to construct the shape function derivatives up to
     the required order
   * Introduce the differential operator putting the differential
     terms together to represent the weak form
   * Introduce the integration template within an element template
   * Introduce the visualization template for an element.
   * Single layer works

Material model extensions
=========================
   * Simplify the DOTSGridEval so that it uses a material parameter.
   * Use a non-linear material model to demonstrate bond.
   * Use a non-linear material model to demonstrate strain-localization
   
Boundary conditions
===================
   * Time function is used in connection with boundary conditions, 
     and parameters of the material behavior.   

Time loop
=========

Response tracers
================
Viz3D system

Visualization
=============
   
   * Introduce units [N/mm]
   * Labels of the diagrams
   * Questions to be answered using this app

   * What is the relation to multi-dimensional case?
   * Generalization
   
   
Time variable
=============
Clarify the meaning of vot and time-line in response tracers
and in boundary conditions.

   * Boundary condition subclasses
   * Run action, pause action, stop action
   * Response tracer for profiles - history tracing 
   * Threaded calculation
   * Efficiency
   * Viz3D sheet
   * Array boundary conditions

@author: rch
'''


from bmcs.time_functions.tfun_pwl_interactive import TFunPWLInteractive
from ibvpy.api import \
    FEGrid, FETSEval, IFETSEval, \
    MATSEval, IMATSEval, \
    TStepper as TS, TLoop, \
    TLine, BCSlice, BCDof, RTDofGraph
from ibvpy.core.i_bcond import IBCond
from ibvpy.dots.dots_grid_eval import DOTSGridEval
from ibvpy.mats.mats1D5.vmats1D5_bondslip1D import MATSBondSlipFatigue
from ibvpy.mesh.i_fe_uniform_domain import IFEUniformDomain
from mathkit.matrix_la import \
    SysMtxAssembly
from traits.api import \
    Int, provides, Array,\
    List, Property, cached_property, Float, \
    Instance, Trait, Button
from traitsui.api import \
    View, Include, Item, UItem, VGroup
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSTreeNode, BMCSLeafNode
from view.window import BMCSModel

import numpy as np
import sympy as sp


n_C = 2

r_ = sp.symbols('r')


@provides(IFETSEval)
class FETS1D52L4ULRH(BMCSLeafNode, FETSEval):
    '''Example of a finite element definition.
    '''

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

    n_ip = Property(depends_on='w_m')

    @cached_property
    def _get_n_ip(self):
        return len(self.w_m)

    tree_view = View(
        VGroup(
            Item('n_e_dofs'),
            Item('n_nodal_dofs'),
        )
    )


class PullOutModelLin(BMCSModel, Vis2D):
    '''Linear elastic calculation of pull-out problem.
    '''

    node_name = 'Pull-out parameters'

    tree_node_list = List

    def _tree_node_list_default(self):
        return [self.fets, self.ts.bcond_mngr, self.ts.rtrace_mngr]

    L_x = Float(1, input=True)

    eta = Float(0.1, input=True)

    G = Float(1.0, bc_changed=True)

    n_E = Int(10, input=True)

    fets = Instance(FETSEval)

    def _fets_default(self):
        return FETS1D52L4ULRH(mats_eval=MATSBondSlipFatigue())

    sdomain = Property(Instance(IFEUniformDomain), depends_on='+input')

    @cached_property
    def _get_sdomain(self):
        return FEGrid(coord_min=(0., ),
                      coord_max=(self.L_x, ),
                      shape=(self.n_E, ),
                      fets_eval=self.fets)


def run_debontrix_lin():
    from view.window import BMCSWindow

    po = PullOutModelLin(fets=FETS1D52L4ULRH(),
                         n_E=5,
                         L_x=1.0,
                         G=1.0)

    w = BMCSWindow(model=po)
    # po.add_viz2d('F-w')
    # po.add_viz2d('field')
    rt = po.ts.rtrace_mngr['Fi,right over w_right']
    rt.add_viz2d('time function')
    w.configure_traits()


if __name__ == '__main__':
    run_debontrix_lin()
