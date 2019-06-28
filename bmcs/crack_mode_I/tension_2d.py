'''
Created on 12.01.2016
@author: RChudoba, ABaktheer, Yingxiong

@todo: derive the size of the state array.
'''

from scipy import interpolate as ip
from traits.api import \
    Property, Instance, cached_property, \
    List, Float, Trait, Int, on_trait_change
from traitsui.api import \
    View, Item, UItem, VGroup

from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import \
    IMATSEval, TLine, BCSlice
from ibvpy.core.bcond_mngr import BCondMngr
from ibvpy.core.vtloop import TimeLoop
from ibvpy.dots.vdots_grid import DOTSGrid
from ibvpy.fets import \
    FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2DElastic, MATS2DMplDamageEEQ, MATS2DScalarDamage, MATS2DMplCSDEEQ  # , MATS2DMplCSDODF
from ibvpy.mats.mats3D.mats3D_sdamage.viz3d_sdamage import Vis3DSDamage,\
    Viz3DSDamage
from ibvpy.mats.mats3D.viz3d_strain_field import \
    Vis3DStrainField, Viz3DStrainField
from ibvpy.mats.mats3D.viz3d_stress_field import \
    Vis3DStressField, Viz3DStressField
import numpy as np
import numpy as np
from simulator.api import Simulator
import traits.api as tr
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSWindow

from .bending3pt_2d import \
    Viz2DForceDeflection, Vis2DCrackBand, CrossSection
from .viz3d_energy import Viz2DEnergy, Vis2DEnergy, Viz2DEnergyReleasePlot


class XCrossSection(BMCSLeafNode):

    '''Parameters of the pull-out cross section
    '''
    node_name = 'cross-section'

    b = Float(50.0,
              CS=True,
              auto_set=False, enter_set=True,
              desc='cross-section width [mm2]')

    traits_view = View(
        VGroup(
            Item('h', full_size=True, resizable=True),
            label='Cross section'
        )
    )

    tree_view = traits_view


class Geometry(BMCSLeafNode):

    node_name = 'geometry'
    H = Float(10.0,
              label='beam depth',
              GEO=True,
              auto_set=False, enter_set=True,
              desc='Depth of the beam')
    L = Float(100.0,
              label='beam length',
              GEO=True,
              auto_set=False, enter_set=True,
              desc='Length of the specimen')
    L_c = Float(4.0,
                GEO=True,
                label='crack band width',
                auto_set=False, enter_set=True,
                desc='Width of the crack band')

    traits_view = View(
        VGroup(
            Item('H'),
            Item('L'),
            Item('L_c'),
            label='Mesh geometry'
        )
    )

    tree_view = traits_view


class TensileTestModel(Simulator):
    '''The tensile test is used to verify the softening behavior
    '''
    node_name = 'tensile test simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.model,
            self.cross_section,
            self.geometry,
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.model,
            self.cross_section,
            self.geometry,
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
    n_e_x = Int(20, MESH=True, auto_set=False, enter_set=True)

    n_e_y = Int(1, MESH=True, auto_set=False, enter_set=True)

    w_max = Float(4, MESH=True, BC=True, auto_set=False, enter_set=True)

    controlled_elem = Property

    def _get_controlled_elem(self):
        return 0

    #=========================================================================
    # Material model
    #=========================================================================
    model_type = Trait('microplane damage (eeg)',
                       {'elastic': MATS2DElastic,
                        'microplane damage (eeq)': MATS2DMplDamageEEQ,
                        'microplane CSD (eeq)': MATS2DMplCSDEEQ,
                        #'microplane CSD (odf)': MATS2DMplCSDODF,
                        'scalar damage': MATS2DScalarDamage
                        },
                       MAT=True
                       )

    @on_trait_change('model_type')
    def _set_model(self):
        self.model = self.model_type_()

    @on_trait_change('BC,MAT,MESH')
    def reset_node_list(self):
        self._update_node_list()

    model = Instance(IMATSEval,
                     MAT=True)
    '''Material model'''

    def _model_default(self):
        return self.model_type_()

    regularization_on = tr.Bool(True, MAT=True, MESH=True,
                                auto_set=False, enter_set=True)

    @on_trait_change('L_cb')
    def broadcast_cb(self):
        if self.regularization_on:
            self.model.omega_fn.L_s = self.L_cb

    #=========================================================================
    # Finite element type
    #=========================================================================
    fets_eval = Property(Instance(FETS2D4Q),
                         depends_on='CS,MAT')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS2D4Q()

    bcond_mngr = Property(Instance(BCondMngr),
                          depends_on='GEO,CS,MAT,BC,MESH')
    '''Boundary condition manager
    '''
    @cached_property
    def _get_bcond_mngr(self):
        bc_list = [self.fixed_y,
                   self.fixed_x,
                   self.control_bc]
        return BCondMngr(bcond_list=bc_list)

    fixed_y = Property(depends_on='CS,BC,MAT,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_y(self):
        return BCSlice(slice=self.fe_grid[0, 0, 0, 0],
                       var='u', dims=[1], value=0)

    fixed_x = Property(depends_on='CS,BC,MAT,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_x(self):
        return BCSlice(slice=self.fe_grid[0, :, 0, :],
                       var='u', dims=[0], value=0)

    control_bc = Property(depends_on='CS,BC,MAT,GEO,MESH')
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return BCSlice(slice=self.fe_grid[-1, :, -1, :],
                       var='u', dims=[0], value=self.w_max)

    dots_grid = Property(Instance(DOTSGrid),
                         depends_on='CS,MAT,GEO,MESH,FE')
    '''Discretization object.
    '''
    @cached_property
    def _get_dots_grid(self):
        dgrid = DOTSGrid(L_x=self.geometry.L, L_y=self.geometry.H,
                         integ_factor=self.cross_section.b,
                         n_x=self.n_e_x, n_y=self.n_e_y,
                         fets=self.fets_eval)

        L = self.geometry.L
        L_c = self.L_cb
        x_x, x_y = dgrid.mesh.geo_grid.point_x_grid
        L_1 = x_x[1, 0]
        d_L = L_c - L_1
        x_x[1:, :] += d_L * (L - x_x[1:, :]) / (L - L_1)
        return dgrid

    fe_grid = Property

    def _get_fe_grid(self):
        return self.dots_grid.mesh

    response_traces = tr.Dict
    '''Response traces.
    '''

    def _response_traces_default(self):
        return {'energy': Vis2DEnergy(model=self),
                }

    t = Property

    def _get_t(self):
        return self.get_t()

    def get_t(self):
        return np.array(self.tloop.t_record, dtype=np.float_)

    def get_PW(self):
        record_dofs = np.unique(
            self.fe_grid[-1, :, -1, :].dofs[:, :, 0].flatten()
        )
        Fd_int_t = np.array(self.tloop.F_int_record)
        Ud_t = np.array(self.tloop.U_record)
        F_int_t = np.sum(Fd_int_t[:, record_dofs], axis=1)
        U_t = Ud_t[:, record_dofs[0]]
        return F_int_t, U_t

    viz2d_classes = {'F-w': Viz2DForceDeflection,
                     'load function': Viz2DLoadControlFunction,
                     }

    traits_view = View(
        Item('w_max'),
        VGroup(
            Item('n_e_x', full_size=True, resizable=True),
            Item('n_e_y'),
            Item('L_cb'),
            label='Numerical parameters'
        ),
        Item('regularization_on'),
        # UItem('geometry@')
    )

    tree_view = traits_view


def run_tensile_test_sdamage(*args, **kw):
    bt = TensileTestModel(n_e_x=20, n_e_y=1, k_max=500,
                          model_type='scalar damage'
                          )
    L = 200.
    L_cb = 5.0
    E = 30000.0
    f_t = 2.4
    G_f = 0.09
    bt.model.trait_set(
        stiffness='algorithmic',
        E=E,
        nu=0.2
    )
    f_t_Em = np.ones_like(bt.dots_grid.state_arrays['omega']) * 10.0
    l_f_t_Em = len(f_t_Em)
    f_t_Em[0, ...] = 1.0
    bt.model.omega_fn.trait_set(
        f_t=f_t,
        f_t_Em=f_t_Em,
        G_f=G_f,
        L_s=L_cb
    )

    bt.w_max = 0.15
    bt.tline.step = 0.01
    bt.cross_section.b = 1
    bt.geometry.trait_set(
        L=L,
        H=1,
        L_c=L_cb
    )
    bt.loading_scenario.trait_set(loading_type='monotonic')
    w = BMCSWindow(model=bt)
    bt.add_viz2d('F-w', 'load-displacement')

    vis2d_energy = bt.response_traces['energy']
    viz2d_energy = Viz2DEnergy(name='dissipation',
                               vis2d=vis2d_energy)
    viz2d_energy_rates = Viz2DEnergyReleasePlot(name='dissipation rate',
                                                vis2d=vis2d_energy)
    w.viz_sheet.viz2d_list.append(viz2d_energy)
    w.viz_sheet.viz2d_list.append(viz2d_energy_rates)
    w.viz_sheet.monitor_chunk_size = 10
    w.viz_sheet.reference_viz2d_name = 'load-displacement'
    w.run()
    w.offline = True
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_tensile_test_sdamage()
