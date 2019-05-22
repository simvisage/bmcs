''' 
Created on 30.04.2019 
 
@author: fseemab 
'''
import time

from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.fets.fets1D5 import FETS1D52ULRH
from ibvpy.mats.mats1D5.vmats1D5_d import \
    MATS1D5D
from ibvpy.mats.mats1D5.vmats1D5_dp import \
    MATS1D5DP
from ibvpy.mats.mats1D5.vmats1D5_dp_cum_press import \
    MATS1D5DPCumPress
from ibvpy.mats.mats2D.mats2D_elastic.vmats2D_elastic import \
    MATS2DElastic
from ibvpy.mats.mats3D.mats3D_elastic.vmats3D_elastic import \
    MATS3DElastic
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.viz2d_field import \
    Vis2DField, Viz2DField
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DTensorField, Viz3DTensorField
from simulator.api import \
    Simulator
from simulator.demo.viz2d_fw import Viz2DFW, Vis2DFW
from simulator.xdomain.xdomain_fe_grid import XDomainFEGrid
from simulator.xdomain.xdomain_interface import XDomainFEInterface
from view.ui.bmcs_tree_node import itags_str
from view.window import BMCSWindow
import traits.api as tr


#from .mlab_decorators import decorate_figure
u_max = 2
#f_max = 30
dx = 1
# ds = 14
r_steel = 1
r_concrete = r_steel * 5
n_x = 1


class PullOutAxiSym(Simulator):

    tree_node_list = tr.List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.m_ifc,
            self.m_steel,
            self.m_concrete,
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.m_ifc,
            self.m_steel,
            self.m_concrete,
        ]

    xd_steel = tr.Property()

    @tr.cached_property
    def _get_xd_steel(self):
        return XDomainFEGrid(coord_min=(0, 0),
                             coord_max=(dx, r_steel),
                             shape=(n_x, 1),
                             integ_factor=1,
                             fets=FETS2D4Q())

    m_steel = tr.Property()

    @tr.cached_property
    def _get_m_steel(self):
        return MATS2DElastic(E=200000, nu=0.3)

    xd_concrete = tr.Property()

    @tr.cached_property
    def _get_xd_concrete(self):
        return XDomainFEGrid(coord_min=(0, r_steel),
                             coord_max=(dx, r_concrete),
                             shape=(n_x, 1),
                             integ_factor=1,
                             fets=FETS2D4Q())

    m_concrete = tr.Property()

    @tr.cached_property
    def _get_m_concrete(self):
        return MATS2DElastic(E=30000, nu=0.2)

    xd_ifc = tr.Property()

    @tr.cached_property
    def _get_xd_ifc(self):
        return XDomainFEInterface(
            I=self.xd_steel.mesh.I[:, -1],
            J=self.xd_concrete.mesh.I[:, 0],
            fets=FETS1D52ULRH(),
            integ_factor=0.5
        )

    m_ifc = tr.Property()

    @tr.cached_property
    def _get_m_ifc(self):
        return MATS1D5DPCumPress(
            E_N=1000000,
            algorithmic=True)  # omega_fn_type='li',

    domains = tr.Property()

    @tr.cached_property
    def _get_domains(self):
        print('domains reconstructed')
        return [
            (self.xd_steel, self.m_steel),
            (self.xd_concrete, self.m_concrete),
            (self.xd_ifc, self.m_ifc),
        ]

    right_x_s = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_right_x_s(self):
        return BCSlice(slice=self.xd_steel.mesh[-1, :, -1, :],
                       var='u', dims=[0], value=u_max)
    right_x_c = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_right_x_c(self):
        return BCSlice(slice=self.xd_concrete.mesh[0, :, 0, :],
                       var='u', dims=[0], value=0)
    left_x_s = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_left_x_s(self):
        return BCSlice(slice=self.xd_steel.mesh[0, :, 0, :],
                       var='f', dims=[0], value=0)

    bc = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_bc(self):
        return [self.right_x_s, self.right_x_c]

    record = {
        'Pw': Vis2DFW(bc_right='right_x_s', bc_left='left_x_s'),
        'Pw2': Vis2DFW(bc_right='right_x_c', bc_left='left_x_s'),
        'slip': Vis2DField(var='slip'),
        'shear': Vis2DField(var='shear'),
        'omega': Vis2DField(var='omega'),
        's_pi': Vis2DField(var='s_pi'),
        's_el': Vis2DField(var='s_el'),
        'alpha': Vis2DField(var='alpha'),
        'z': Vis2DField(var='z'),
        'strain': Vis3DTensorField(var='eps_ab'),
        'stress': Vis3DTensorField(var='sig_ab'),
        #        'damage': Vis3DStateField(var='omega_a'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }

    def get_window(self):

        fw = Viz2DFW(name='Pw', vis2d=self.hist['Pw'])
        fw2 = Viz2DFW(name='Pw2', vis2d=self.hist['Pw2'])
        fslip = Viz2DField(name='slip', vis2d=self.hist['slip'])
        fshear = Viz2DField(name='shear', vis2d=self.hist['shear'])
        fomega = Viz2DField(name='omega', vis2d=self.hist['omega'])
        fs_pi = Viz2DField(name='s_pi', vis2d=self.hist['s_pi'])
        fs_el = Viz2DField(name='s_el', vis2d=self.hist['s_el'])
        falpha = Viz2DField(name='alpha', vis2d=self.hist['alpha'])
        fz = Viz2DField(name='z', vis2d=self.hist['z'])

        w = BMCSWindow(sim=self)
        w.viz_sheet.viz2d_list.append(fw)
        w.viz_sheet.viz2d_list.append(fw2)
        w.viz_sheet.viz2d_list.append(fslip)
        w.viz_sheet.viz2d_list.append(fs_el)
        w.viz_sheet.viz2d_list.append(fs_pi)
        w.viz_sheet.viz2d_list.append(fshear)
        w.viz_sheet.viz2d_list.append(fomega)
        w.viz_sheet.viz2d_list.append(falpha)
        w.viz_sheet.viz2d_list.append(fz)
        strain_viz = Viz3DTensorField(vis3d=s.hist['strain'])
        w.viz_sheet.add_viz3d(strain_viz)
        stress_viz = Viz3DTensorField(vis3d=s.hist['stress'])
        w.viz_sheet.add_viz3d(stress_viz)
        return w


s = PullOutAxiSym()
s.m_ifc.trait_set(E_T=12000,
                  tau_bar=4.0,
                  K=0, gamma=10,
                  c=1, S=0.025, r=1)
s.tloop.k_max = 1000
s.tloop.verbose = True
s.tline.step = 0.005
s.tstep.fe_domain.serialized_subdomains
w = s.get_window()
w.configure_traits()
