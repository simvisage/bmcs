'''
Created on May 15, 2018

@author: rch
'''

'''
Created on 12.01.2016
@author: RChudoba, ABaktheer, Yingxiong

@todo: derive the size of the state array.
'''

from traits.api import \
    Property, Instance, cached_property, \
    List, Float, Trait, Int, on_trait_change
from traitsui.api import \
    View, Item
from ibvpy.api import \
    BCSlice, BCDof
from ibvpy.dots.vdots_grid3d import DOTSGrid
from ibvpy.fets import FETS3D8H
from ibvpy.mats.mats3D import \
    MATS3DMplDamageODF, MATS3DMplDamageEEQ, MATS3DElastic, \
    MATS3DScalarDamage
from ibvpy.mats.mats3D.mats3D_sdamage.viz3d_sdamage import \
    Vis3DSDamage, Viz3DSDamage
import numpy as np
from simulator import Simulator, Model
import traits.api as tr
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSWindow


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
    L = Float(20.0,
              GEO=True,
              auto_set=False, enter_set=True,
              desc='Length of the specimen')

    view = View(
        Item('L'),
    )

    tree_view = view


class BendingTestSimulator(Simulator):

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
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.mats_eval,
            self.cross_section,
            self.geometry,
        ]

    @on_trait_change('BC,MAT,MESH')
    def reset_node_list(self):
        self._update_node_list()

    cross_section = Instance(CrossSection)

    def _cross_section_default(self):
        return CrossSection()

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    #=========================================================================
    # Discretization
    #=========================================================================
    n_e_x = Int(2, auto_set=False, enter_set=True)
    n_e_y = Int(8, auto_set=False, enter_set=True)
    n_e_z = Int(1, auto_set=False, enter_set=True)

    w_max = Float(0.01, BC=True, auto_set=False, enter_set=True)

    #=========================================================================
    # Material model
    #=========================================================================
    model_type = Trait('scalar damage',
                       {'elastic': MATS3DElastic,
                        'scalar damage': MATS3DScalarDamage,
                        'microplane damage (eeq)': MATS3DMplDamageEEQ,
                        'microplane damage (odf)': MATS3DMplDamageODF,
                        },
                       MAT=True
                       )

    @on_trait_change('model_type')
    def _set_model(self):
        self.model = self.model_type_()

    model = Instance(Model,
                     MAT=True)
    '''Material model'''

    def _model_default(self):
        return self.model_type_()

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

    def _get_bc(self):
        return [self.fixed_left_x,
                self.fixed_left_y,
                self.fixed_left_z,
                # self.fixed_top_right_x,
                self.link_right_cs,
                self.control_bc,
                ] + self.link_right_x

    fixed_left_x = Property(depends_on='CS, BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_left_x(self):
        return BCSlice(slice=self.fe_grid[0, :, :, 0, -1, :],
                       var='u', dims=[0], value=0)

    fixed_left_y = Property(depends_on='CS, BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_left_y(self):
        return BCSlice(slice=self.fe_grid[0, -1, :, 0, -1, :],
                       var='u', dims=[1], value=0)

    fixed_left_z = Property(depends_on='CS, BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_left_z(self):
        return BCSlice(slice=self.fe_grid[0, -1, 0, 0, -1, 0],
                       var='u', dims=[2], value=0)

    link_right_cs = Property(depends_on='CS,BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_link_right_cs(self):
        f_dof = self.fe_grid[-1, :, -1, -1, :, -1]
        b_dof = self.fe_grid[-1, :, 0, -1, :, 0]
        return BCSlice(name='link_cs', slice=f_dof, link_slice=b_dof, dims=[0],
                       link_coeffs=[1], value=0)

    link_right_x = Property(depends_on='CS,BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_link_right_x(self):
        top = self.fe_grid[-1, -1, 0, -1, -1, 0]
        bot = self.fe_grid[-1, 0, 0, -1, 0, 0]
        linked = self.fe_grid[-1, 1:, 0, -1, 0, 0]

        Ty = top.dof_X[0, 0, 1]
        By = bot.dof_X[0, 0, 1]

        Ly = linked.dof_X[:, :, 1].flatten()

        H = Ty - By
        link_ratios = Ly / H
        top_dof = top.dofs[0, 0, 0]
        bot_dof = bot.dofs[0, 0, 0]
        linked_dofs = linked.dofs[:, :, 0].flatten()
        bcdof_list = []
        for linked_dof, link_ratio in zip(linked_dofs, link_ratios):
            link_bc = BCDof(var='u',
                            dof=linked_dof,
                            value=0,
                            link_dofs=[bot_dof, top_dof],
                            link_coeffs=[1 - link_ratio, link_ratio]
                            )
            bcdof_list.append(link_bc)
        return bcdof_list

    fixed_top_right_x = Property(depends_on='CS,BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_top_right_x(self):
        return BCSlice(slice=self.fe_grid[-1, -1, 0, -1, -1, 0],
                       var='u', dims=[0], value=0)

    control_bc = Property(depends_on='CS,BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_control_bc(self):
        return BCSlice(
            slice=self.fe_grid[-1, 0, 0, -1, 0, 0],
            var='u', dims=[0], value=self.w_max
        )

    dots_grid = Property(Instance(DOTSGrid),
                         depends_on='CS,MAT,GEO,MESH,FE')
    '''Discretization object.
    '''
    @cached_property
    def _get_dots_grid(self):
        cs = self.cross_section
        geo = self.geometry
        return DOTSGrid(
            L_x=geo.L, L_y=cs.h, L_z=cs.b,
            n_x=self.n_e_x, n_y=self.n_e_y, n_z=self.n_e_z,
            fets=self.fets_eval, mats=self.mats_eval
        )

    fe_grid = Property

    def _get_fe_grid(self):
        return self.dots_grid.mesh

    def get_PW(self):
        record_dofs = self.fe_grid[
            -1, 0, :, -1, 0, :].dofs[:, :, 0].flatten()
        Fd_int_t = np.array(self.tloop.F_int_record)
        Ud_t = np.array(self.tloop.U_record)
        F_int_t = np.sum(Fd_int_t[:, record_dofs], axis=1)
        U_t = Ud_t[:, record_dofs[0]]
        return F_int_t, U_t

    traits_view = View(Item('model_type'),)

    tree_view = traits_view


def run_bending3pt_mic_odf(*args, **kw):

    bt = BendingTestSimulator(
        n_e_x=2, n_e_y=10, n_e_z=1,
        k_max=500,
        mats_eval_type='scalar damage'
        #mats_eval_type='microplane damage (odf)'
    )
    bt.mats_eval.trait_set(
        # stiffness='algorithmic',
        epsilon_0=59e-6,
        epsilon_f=600e-6
    )

    bt.w_max = 0.02
    bt.tline.step = 0.05
    bt.cross_section.h = 100
    bt.cross_section.b = 50
    bt.geometry.L = 20
    bt.loading_scenario.trait_set(loading_type='monotonic')
    w = BMCSWindow(model=bt)
    bt.add_viz2d('load function', 'load-time')
    bt.add_viz2d('F-w', 'load-displacement')

    vis3d = Vis3DSDamage()
    bt.tloop.response_traces.append(vis3d)
    viz3d = Viz3DSDamage(vis3d=vis3d)
    w.viz_sheet.add_viz3d(viz3d)
    w.viz_sheet.monitor_chunk_size = 1

#    w.run()
    w.offline = False
#    w.finish_event = True
    w.configure_traits()


if __name__ == '__main__':
    run_bending3pt_mic_odf()
