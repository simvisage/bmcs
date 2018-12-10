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


from traits.api import \
    Int, provides, Array,\
    List, Property, cached_property, Float, \
    Instance, Trait, Button
from traitsui.api import \
    View, Include, Item, UItem, VGroup

from bmcs.mats.mats_bondslip import MATSBondSlipFatigue
from bmcs.time_functions.tfun_pwl_interactive import TFunPWLInteractive
from ibvpy.api import \
    FEGrid, FETSEval, IFETSEval, \
    MATSEval, IMATSEval, \
    TStepper as TS, TLoop, \
    TLine, BCSlice, BCDof, RTDofGraph
from ibvpy.core.i_bcond import IBCond
from ibvpy.dots.dots_grid_eval import DOTSGridEval
from ibvpy.mesh.i_fe_uniform_domain import IFEUniformDomain
from mathkit.matrix_la import \
    SysMtxAssembly
import numpy as np
import sympy as sp
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSTreeNode, BMCSLeafNode
from view.window import BMCSModel


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

    node_name = 'Finite element parameters'

    def get_eps(self, dN_Eimd, sN_Cim, U_ECid, dU_ECid):
        eps1 = np.einsum('Emid,ECie->ECmde', dN_Eimd, U_ECid)
        eps2 = np.einsum('Emie,ECid->ECmde', dN_Eimd, U_ECid)
        eps_ECmde = (eps1 + eps2) / 2.0
        s_ECmd = np.einsum('Cim,ECid->ECmd', sN_Cim, U_ECid)
        return eps_ECmde, s_ECmd

    tree_view = View(
        VGroup(
            Item('n_e_dofs'),
            Item('n_nodal_dofs'),
        )
    )


class Viz2DPullOutFW(Viz2D):
    '''Plot adaptor for the force-displacement curve.
    '''

    w = List([])
    Fint = List([])

    def plot(self, ax, vot, *args, **kw):
        d = self.vis2d.d_IC[-1, -1]
        Fint = self.vis2d.Fint_IC[-1, -1]
        self.w.append(d)
        self.Fint.append(Fint)
        ax.plot(self.w, self.Fint)

    clear = Button()

    def _clear_fired(self):
        self.w = []
        self.Fint = []

    traits_view = View(
        UItem('clear')
    )


class Viz2DPullOutField(Viz2D):
    '''Plot adaptor for the pull-out simulator.

    What the difference between the response tracer
    and visualization adapter?

    Response tracer gathers the data during the computation.
    It maps the recorded data to the time axis governing
    the calculation.

    Visualization adapter (Viz2D, Viz3D) is used to
    to map the recorded data into the interface. 
    Visualization adapter can acquire the data from
    input objects prescribing e.g. the boundary conditions
    or geometry of the boundary value problem.

    It can also use a response tracer to map the calculated
    time dependency to the user interface. 
    '''
    label = Property(depends_on='plot_fn')

    @cached_property
    def _get_label(self):
        return 'field: %s' % self.plot_fn

    plot_fn = Trait('eps_C',
                    {'eps_C': 'plot_eps_C',
                     'u_C': 'plot_u_C',
                     's': 'plot_s',
                     'Fint_C': 'plot_Fint_C'
                     },
                    label='Field',
                    tooltip='Select the field to plot'
                    )

    def plot(self, ax, vot, *args, **kw):
        #         w_max = self.vis2d.w_max
        #         self.vis2d.w = vot * w_max
        getattr(self.vis2d, self.plot_fn_)(ax)

    traits_view = View(
        Item('plot_fn')
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

    dots = Property(Instance(DOTSGridEval), depends_on='+input')

    @cached_property
    def _get_dots(self):
        return DOTSGridEval(fets_eval=self.fets,
                            sdomain=self.sdomain,
                            eta=self.eta,
                            G=self.G
                            )

    bc = Property(Instance(IBCond), depends_on='+input')

    @cached_property
    def _get_bc(self):
        dof = self.sdomain[-1, -1].dofs[0, 0, 1]
        tfun = TFunPWLInteractive()
        return BCDof(var='u', dof=dof, value=0.001,
                     time_function=tfun
                     )

    ts = Property(Instance(TS),
                  depends_on='+input,+bc_changed')

    @cached_property
    def _get_ts(self):
        return TS(dof_resultants=True,
                  node_name='Pull-out',
                  tse=self.dots,
                  sdomain=self.sdomain,
                  bcond_list=[
                      BCSlice(var='u', value=0., dims=[0],
                              slice=self.sdomain[0, 0]),
                      self.bc,
                  ],
                  rtrace_list=[RTDofGraph(name='Fi,right over w_right',
                                          var_y='F_int', idx_y=-1, cum_y=True,
                                          var_x='U_k', idx_x=-1),
                               ]
                  )

    tloop = Property(Instance(TLoop),
                     depends_on='+input,+bc_changed,tmax_changed')
    '''Time loop control.
    '''
    @cached_property
    def _get_tloop(self):
        return TLoop(tstepper=self.ts, KMAX=30,
                     debug=False, tline=self.tline
                     )

    tline = Instance(TLine)
    '''Time range.
    '''

    def _tline_default(self):
        return TLine(min=0.0, step=0.1, max=0.0,
                     time_change_notifier=self.time_changed,
                     )

    def init(self):
        self.tloop.init()

    def eval(self):
        return self.tloop.eval()

    def pause(self):
        self.tloop.paused = True

    def stop(self):
        self.tloop.restart = True

    K_Eij = Property(depends_on='+input,+bc_changed')

    @cached_property
    def _get_K_Eij(self):
        fet = self.fets
        K_ECidDjf = self.dots.BB_ECidDjf + self.dots.NN_ECidDjf * self.G
        K_Eij = K_ECidDjf.reshape(-1, fet.n_e_dofs, fet.n_e_dofs)
        return K_Eij

    dd = Property(
        depends_on='+input,+bc_changed,+time_changed')

    @cached_property
    def _get_dd(self):
        n_dof_tot = self.sdomain.n_dofs
        # System matrix
        K = SysMtxAssembly()
        K.add_mtx_array(self.K_Eij, self.dots.dof_E)
        K.register_constraint(0, 0.0)
        K.register_constraint(n_dof_tot - 1, self.w)
        F_ext = np.zeros((n_dof_tot,), np.float_)
        K.apply_constraints(F_ext)
        d = K.solve(F_ext)
        return d

    d = Property(Array(np.float),
                 depends_on='+input,+bc_changed,+time_changed')

    @cached_property
    def _get_d(self):
        return self.tloop.eval()

    d_IC = Property

    def _get_d_IC(self):
        d_ECid = self.d[self.dots.dof_ECid]
        return np.einsum('ECid->EidC', d_ECid).reshape(-1, n_C)

    eps_C = Property

    def _get_eps_C(self):
        d_ECid = self.d[self.dots.dof_ECid]
        eps_EmdC = np.einsum('Eimd,ECid->EmdC', self.dots.dN_Eimd, d_ECid)
        return eps_EmdC.reshape(-1, n_C)

    u_C = Property
    '''Displacement field
    '''

    def _get_u_C(self):
        d_ECid = self.d[self.dots.dof_ECid]
        N_mi = self.fets.N_mi
        u_EmdC = np.einsum('mi,ECid->EmdC', N_mi, d_ECid)
        return u_EmdC.reshape(-1, n_C)

    s = Property
    '''Slip between the two material phases'''

    def _get_s(self):
        d_ECid = self.d[self.dots.dof_ECid]
        s_Emd = np.einsum('Cim,ECid->Emd', self.dots.sN_Cim, d_ECid)
        return s_Emd.flatten()

    Fint_I = Property

    def _get_Fint_I(self):
        K_ECidDjf = self.dots.BB_ECidDjf + self.dots.NN_ECidDjf * self.G
        d_ECid = self.d[self.dots.dof_ECid]
        f_ECid = np.einsum('ECidDjf,EDjf->ECid', K_ECidDjf, d_ECid)
        f_Ei = f_ECid.reshape(-1, self.fets.n_e_dofs)
        return np.bincount(self.dots.dof_E.flatten(), weights=f_Ei.flatten())

    Fint_IC = Property

    def _get_Fint_IC(self):
        return self.Fint_I.reshape(-1, n_C)

    def plot_Fint_C(self, ax):
        ax.plot(self.dots.X_Id.flatten(), self.Fint_IC)

    def plot_u_C(self, ax):
        ax.plot(self.dots.X_J, self.u_C)

    def plot_eps_C(self, ax):
        ax.plot(self.dots.X_M, self.eps_C)

    def plot_s(self, ax):
        ax.plot(self.dots.X_J, self.s)

    def plot(self, fig):
        ax = fig.add_subplot(221)
        self.plot_Fint_C(ax)
        ax = fig.add_subplot(222)
        self.plot_eps_C(ax)
        ax = fig.add_subplot(223)
        self.plot_s(ax)
        ax = fig.add_subplot(224)
        self.plot_u_C(ax)

    tree_view = View(
        Include('actions'),
        Item('n_E', label='Number of elements'),
        Item('G', label='Shear modulus'),
        Item('eta', label='E_m/E_f'),
        Item('L_x', label='Specimen length')
    )

    viz2d_classes = {'field': Viz2DPullOutField,
                     'F-w': Viz2DPullOutFW}


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
