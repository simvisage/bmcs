'''
Created on 12.01.2016
@author: RChudoba, ABaktheer

'''
import time
from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import \
    IMATSEval, BCSlice
from ibvpy.core.bcond_mngr import BCondMngr
from ibvpy.fets import FETS3D8H
from ibvpy.mats.mats3D import \
    MATS3DMplDamageODF, MATS3DMplDamageEEQ, MATS3DElastic, \
    MATS3DScalarDamage
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DTensorField, Viz3DTensorField
from simulator.api import Simulator, XDomainFEGrid
from traits.api import \
    Property, Instance, cached_property, \
    List, Float, Trait, Int, on_trait_change
from traitsui.api import \
    View, Item
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.ui.bmcs_tree_node import itags_str
from view.window import BMCSWindow

import numpy as np
import traits.api as tr


class Viz2DForceDeflectionX(Viz2D):

    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        P, W = self.vis2d.get_PW()
        ymin, ymax = np.min(P), np.max(P)
        L_y = ymax - ymin
        ymax += 0.05 * L_y
        ymin -= 0.05 * L_y
        xmin, xmax = np.min(W), np.max(W)
        L_x = xmax - xmin
        xmax += 0.03 * L_x
        xmin -= 0.03 * L_x
        ax.plot(W, P, linewidth=2, color='black', alpha=0.4,
                label='P(w;x=L)')
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylabel('Force P [N]')
        ax.set_xlabel('Deflection w [mm]')
        if self.show_legend:
            ax.legend(loc=4)
        self.plot_marker(ax, vot)

    def plot_marker(self, ax, vot):
        P, W = self.vis2d.get_PW()
        idx = self.vis2d.tloop.get_time_idx(vot)
        P, w = P[idx], W[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)

    def plot_tex(self, ax, vot, *args, **kw):
        self.plot(ax, vot, *args, **kw)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )


class CrossSection(BMCSLeafNode):

    '''Parameters of the pull-out cross section
    '''
    node_name = 'cross-section'

    b = Float(50.0,
              CS=True,
              auto_set=False, enter_set=True,
              desc='cross-section width [mm2]')
    h = Float(100.0,
              CS=True,
              auto_set=False, enter_set=True,
              desc='cross section height [mm2]')

    view = View(
        Item('h'),
        Item('b'),
    )

    tree_view = view


class Geometry(BMCSLeafNode):

    node_name = 'geometry'
    L = Float(600.0,
              GEO=True,
              auto_set=False, enter_set=True,
              desc='Length of the specimen')

    view = View(
        Item('L_x'),
    )

    tree_view = view


class BendingTestModel(Simulator, Vis2D):

    #=========================================================================
    # Tree node attributes
    #=========================================================================
    node_name = 'bending test simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.mats_eval,
            self.cross_section,
            self.geometry,
            self.fixed_left_bc,
            self.fixed_right_bc,
            self.control_bc
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.mats_eval,
            self.cross_section,
            self.geometry,
            self.fixed_left_bc,
            self.fixed_right_bc,
            self.control_bc
        ]

    #=========================================================================
    # Test setup parameters
    #=========================================================================
    loading_scenario = Instance(LoadingScenario)

    def _loading_scenario_default(self):
        return LoadingScenario()

    cross_section = Instance(CrossSection)

    def _cross_section_default(self):
        return CrossSection()

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    #=========================================================================
    # Discretization
    #=========================================================================
    n_e_x = Int(20, auto_set=False, enter_set=True)
    n_e_y = Int(8, auto_set=False, enter_set=True)
    n_e_z = Int(1, auto_set=False, enter_set=True)

    w_max = Float(-50, BC=True, auto_set=False, enter_set=True)

    controlled_elem = Property(Int)

    def _get_controlled_elem(self):
        return int(self.n_e_x / 2)

    #=========================================================================
    # Material model
    #=========================================================================
    mats_eval_type = Trait('microplane damage (eeq)',
                           {'elastic': MATS3DElastic,
                            'microplane damage (eeq)': MATS3DMplDamageEEQ,
                            'microplane damage (odf)': MATS3DMplDamageODF,
                            'scalar damage': MATS3DScalarDamage,
                            },
                           MAT=True
                           )

    @on_trait_change('mats_eval_type')
    def _set_mats_eval(self):
        self.mats_eval = self.mats_eval_type_()

    @on_trait_change('BC,MAT,MESH')
    def reset_node_list(self):
        self._update_node_list()

    mats_eval = Instance(IMATSEval,
                         MAT=True)
    '''Material model'''

    def _mats_eval_default(self):
        return self.mats_eval_type_()

    material = Property

    def _get_material(self):
        return self.mats_eval

    #=========================================================================
    # Finite element type
    #=========================================================================
    fets_eval = Property(Instance(FETS3D8H),
                         depends_on='CS,MAT')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS3D8H()

    bc = Property(Instance(BCondMngr),
                  depends_on='GEO,CS,BC,MAT,MESH')
    '''Boundary condition manager
    '''
    @cached_property
    def _get_bc(self):
        return [self.fixed_left_bc,
                self.fixed_right_bc,
                self.fixed_middle_bc,
                self.control_bc]

    fixed_left_bc = Property(depends_on='CS, BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_left_bc(self):
        return BCSlice(slice=self.fe_grid[0, 0, :, 0, 0, :],
                       var='u', dims=[1], value=0)

    fixed_right_bc = Property(depends_on='CS,BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_right_bc(self):
        return BCSlice(slice=self.fe_grid[-1, 0, :, -1, 0, :],
                       var='u', dims=[1], value=0)

    fixed_middle_bc = Property(depends_on='CS,BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_middle_bc(self):
        return BCSlice(
            slice=self.fe_grid[self.controlled_elem, -1, :, 0, -1, :],
            var='u', dims=[0], value=0
        )

    control_bc = Property(depends_on='CS,BC,GEO,MESH')
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return BCSlice(
            slice=self.fe_grid[self.controlled_elem, -1, :, :, -1, :],
            var='u', dims=[1], value=-self.w_max
        )

    xdomain = Property(depends_on='CS,MAT,GEO,MESH,FE')
    '''Discretization object.
    '''
    @cached_property
    def _get_xdomain(self):
        cs = self.cross_section
        geo = self.geometry
        dgrid = XDomainFEGrid(dim_u=3,
                              coord_max=(geo.L, cs.h, cs.b),
                              shape=(self.n_e_x, self.n_e_y, self.n_e_z),
                              fets=self.fets_eval)
        return dgrid
        L = self.geometry.L / 2.0
        L_c = self.geometry.L_c
        x_x, _, _ = dgrid.mesh.geo_grid.point_x_grid
        L_1 = x_x[1, 0]
        d_L = L_c - L_1
        x_x[1:, :, :] += d_L * (L - x_x[1:, :]) / (L - L_1)
        return dgrid

    fe_grid = Property

    def _get_fe_grid(self):
        return self.xdomain.mesh

    domains = Property(depends_on=itags_str)

    @cached_property
    def _get_domains(self):
        return [(self.xdomain, self.mats_eval)]

    k_max = Int(200,
                ALG=True)
    acc = Float(1e-4,
                ALG=True)

    @on_trait_change('ALG')
    def _reset_tloop(self):
        k_max = self.k_max
        acc = self.acc
        self.tloop.trait_set(
            k_max=k_max,
            acc=acc,
        )

    def get_PW(self):
        record_dofs = self.fe_grid[
            self.controlled_elem, -1, :, :, -1, :].dofs[:, :, 1].flatten()
        Fd_int_t = np.array(self.tloop.F_int_record)
        Ud_t = np.array(self.tloop.U_record)
        F_int_t = -np.sum(Fd_int_t[:, record_dofs], axis=1)
        U_t = -Ud_t[:, record_dofs[0]]
        return F_int_t, U_t

    viz2d_classes = {'F-w': Viz2DForceDeflectionX,
                     'load function': Viz2DLoadControlFunction,
                     }

    traits_view = View(Item('mats_eval_type'),)

    tree_view = traits_view


def run_bending3pt_mic_odf(*args, **kw):

    bt = BendingTestModel(n_e_x=21, n_e_y=5, n_e_z=1,
                          k_max=500,
                          mats_eval_type='microplane damage (eeq)'
                          #mats_eval_type='microplane damage (eeq)'
                          #mats_eval_type='microplane damage (odf)'
                          )
    E_c = 28000  # MPa
    f_ct = 3.0  # MPa
    epsilon_0 = f_ct / E_c  # [-]

    print(bt.mats_eval_type)
    bt.mats_eval.trait_set(
        # stiffness='algorithmic',
        epsilon_0=epsilon_0,
        epsilon_f=epsilon_0 * 3
    )

    bt.w_max = 1
    bt.tline.step = 0.02
    bt.cross_section.h = 100
    bt.geometry.L = 800
    bt.loading_scenario.trait_set(loading_type='monotonic')

    bt.record = {
        #       'Pw': Vis2DFW(bc_right=right_x_s, bc_left=left_x_s),
        #       'slip': Vis2DField(var='slip'),
        'strain': Vis3DTensorField(var='eps_ab'),
        'stress': Vis3DTensorField(var='sig_ab'),
        'damage': Vis3DTensorField(var='phi_ab'),
    }

    w = BMCSWindow(sim=bt)
#    bt.add_viz2d('load function', 'load-time')
#    bt.add_viz2d('F-w', 'load-displacement')

    viz_stress = Viz3DTensorField(vis3d=bt.hist['strain'])
    viz_strain = Viz3DTensorField(vis3d=bt.hist['stress'])
    viz_damage = Viz3DTensorField(vis3d=bt.hist['damage'])

    w.viz_sheet.add_viz3d(viz_stress)
    w.viz_sheet.add_viz3d(viz_strain)
    w.viz_sheet.add_viz3d(viz_damage)
    w.viz_sheet.monitor_chunk_size = 1

    w.run()
    time.sleep(10)
    w.offline = False
#    w.finish_event = True
    w.configure_traits()


if __name__ == '__main__':
    run_bending3pt_mic_odf()
