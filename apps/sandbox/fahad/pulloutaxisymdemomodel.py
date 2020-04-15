'''
Created on 20 Nov 2019

@author: fseemab
'''
import time

from ibvpy.bcond import BCSlice
from ibvpy.bcond.bc_dof import BCDof
from ibvpy.fets import FETS2D4Q
from ibvpy.fets.fets1D5 import FETS1D52ULRH
from ibvpy.mats.mats1D5.vmats1D5_d import \
    MATS1D5D
from ibvpy.mats.mats1D5.vmats1D5_dp import \
    MATS1D5DP
from ibvpy.mats.mats1D5.vmats1D5_dp_cum_press import \
    MATS1D5DPCumPress
from ibvpy.mats.mats3D.mats3D_elastic.vmats3D_elastic import \
    MATS3DElastic
from ibvpy.mats.mats3D.mats3D_microplane.vmats3D_mpl_csd_odf import \
    MATS3DMplCSDODF
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.mats3D.mats3D_sdamage.vmats3D_sdamage import \
    MATS3DScalarDamage
from ibvpy.mats.viz2d_field import \
    Vis2DField, Viz2DField
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DTensorField, Viz3DTensorField
from mathkit.mfn import MFnLineArray
from reporter import RInputRecord
from simulator.api import \
    Simulator
from simulator.demo.viz2d_fw import Viz2DFW, Vis2DFW
from simulator.xdomain.xdomain_fe_grid_axisym import XDomainFEGridAxiSym
from simulator.xdomain.xdomain_interface import XDomainFEInterface
from view.ui import BMCSLeafNode
from view.ui.bmcs_tree_node import itags_str
from view.window import BMCSWindow

from apps.sandbox.fahad.vmats1d5_dp_new import \
    MATS1D5DPCumPressnew
from apps.sandbox.fahad.new2dmatmodel import MATS1D5DP2D
import numpy as np
import pylab as p
import traits.api as tr
import traitsui.api as ui


class CrossSection(BMCSLeafNode, RInputRecord):
    '''Parameters of the pull-out cross section
    '''
    node_name = 'cross-section'

    R_m = tr.Float(20,
                   CS=True,
                   input=True,
                   unit=r'$\mathrm{mm}$',
                   symbol=r'R_\mathrm{m}',
                   auto_set=False, enter_set=True,
                   desc='matrix area')
    R_f = tr.Float(1.0,
                   CS=True,
                   input=True,
                   unit='$\\mathrm{mm}$',
                   symbol='R_\mathrm{f}',
                   auto_set=False, enter_set=True,
                   desc='reinforcement area')
    P_b = tr.Property(unit='$\\mathrm{mm}$',
                      symbol='p_\mathrm{b}',
                      desc='perimeter of the bond interface',
                      depends_on='R_f')

    @tr.cached_property
    def _get_P_b(self):
        return 2 * np.pi * self.R_f

    view = ui.View(
        ui.Item('R_m'),
        ui.Item('R_f'),
        ui.Item('P_b', style='readonly')
    )

    tree_view = view


class Geometry(BMCSLeafNode, RInputRecord):

    node_name = 'geometry'
    L_x = tr.Float(45,
                   GEO=True,
                   input=True,
                   unit='$\mathrm{mm}$',
                   symbol='L',
                   auto_set=False, enter_set=True,
                   desc='embedded length')

    view = ui.View(
        ui.Item('L_x'),
    )

    tree_view = view


class PullOutAxiSym(Simulator):

    tree_node_list = tr.List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.geometry,
            self.cross_section,
            self.m_ifc,
            self.m_steel,
            self.m_concrete,
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.geometry,
            self.cross_section,
            self.m_ifc,
            self.m_steel,
            self.m_concrete,
        ]

    u_max = tr.Float(BC=True, auto_set=False, enter_set=True)
    '''Radius of the pullout test
    '''
    
    cross_section = tr.Instance(
        CrossSection,
        report=True,
        desc='cross section parameters'
    )

    def _cross_section_default(self):
        return CrossSection()

    geometry = tr.Instance(
        Geometry,
        report=True,
        desc='geometry parameters of the boundary value problem'
    )

    def _geometry_default(self):
        return Geometry()

    n_x = tr.Int(20,
                 MESH=True,
                 auto_set=False,
                 enter_set=True,
                 symbol='n_\mathrm{E}',
                 unit='-',
                 desc='number of finite elements along the embedded length'
                 )

    n_y_concrete = tr.Int(1,
                          MESH=True,
                          auto_set=False,
                          enter_set=True,
                          symbol='n_\mathrm{E}',
                          unit='-',
                          desc='number of finite elements along concrete radius'
                          )
    n_y_steel = tr.Int(1,
                       MESH=True,
                       auto_set=False,
                       enter_set=True,
                       symbol='n_\mathrm{E}',
                       unit='-',
                       desc='number of finite elements along steel radius'
                       )

    xd_steel = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_xd_steel(self):
        dx = self.geometry.L_x
        r_steel = self.cross_section.R_f
        return XDomainFEGridAxiSym(coord_min=(0, 0),
                                   coord_max=(dx, r_steel),
                                   shape=(self.n_x, self.n_y_steel),
                                   integ_factor=2 * np.pi,
                                   fets=FETS2D4Q())

    m_steel = tr.Instance(MATS3DElastic)

    def _m_steel_default(self):
        return MATS3DElastic(E=200000, nu=0.3)

    xd_concrete = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_xd_concrete(self):
        r_steel = self.cross_section.R_f
        print('len', self.geometry.L_x)
        dx = self.geometry.L_x
        r_concrete = self.cross_section.R_m
        return XDomainFEGridAxiSym(coord_min=(0, r_steel),
                                   coord_max=(dx, r_concrete),
                                   shape=(self.n_x, self.n_y_concrete),
                                   integ_factor=2 * np.pi,
                                   fets=FETS2D4Q())

    m_concrete = tr.Instance(MATS3DElastic)

    def _m_concrete_default(self):
        return MATS3DElastic(E=30000, nu=0.2)

    xd_ifc = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_xd_ifc(self):
        return XDomainFEInterface(
            I=self.xd_steel.mesh.I[:, -1],
            J=self.xd_concrete.mesh.I[:, 0],
            fets=FETS1D52ULRH(),
            integ_factor=self.cross_section.P_b
        )

    m_ifc = tr.Instance( MATS1D5DPCumPress)

    def _m_ifc_default(self):
        return  MATS1D5DPCumPress(
            E_T=10000,
            E_N=1000000,
            gamma=55.0,
            K=11.0,
            tau_bar=4.2,
            S=0.005,
            r=1.0,
            c=2.8,
            m=0.175,
            algorithmic=True)  # omega_fn_type='li',

    domains = tr.Property(depends_on=itags_str)

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
                       var='u', dims=[0], value=self.u_max)

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

    bc_y_0 = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_bc_y_0(self):
        return BCSlice(slice=self.xd_steel.mesh[:, -1, :, -1],
                       var='u', dims=[1], value=0)

    f_lateral = tr.Float(0, auto_set=False, enter_set=True, BC=True)

    bc_lateral_pressure_dofs = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_bc_lateral_pressure_dofs(self):
        tf = MFnLineArray(xdata=[0, 1], ydata=[1, 1])
        mesh_slice = self.xd_concrete.mesh[:, -1, :, -1]
        dofs = np.unique(mesh_slice.dofs[:, :, 1].flatten())
        return [BCDof(dof=dof,
                      var='f', value=self.f_lateral, time_function=tf)
                for dof in dofs]

    bc = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_bc(self):
        self.bc_lateral_pressure_dofs
        return [self.right_x_s, self.right_x_c, self.bc_y_0] + \
            self.bc_lateral_pressure_dofs  # 
 
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
        #----------------------------- 'damage': Vis3DStateField(var='omega_a'),
               #------------ # 'kinematic hardening': Vis3DStateField(var='z_a')
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
        strain_viz = Viz3DTensorField(vis3d=self.hist['strain'])
        w.viz_sheet.add_viz3d(strain_viz)
        stress_viz = Viz3DTensorField(vis3d=self.hist['stress'])
        w.viz_sheet.add_viz3d(stress_viz)
        return w

    tree_view = ui.View(
        ui.Item('u_max'),
        ui.Item('n_x'),
        ui.Item('n_y_steel'),
        ui.Item('n_y_concrete'),
        ui.Item('f_lateral'),
    )


if __name__ == '__main__':
    s = PullOutAxiSym(u_max=0.04, f_lateral=-100)
    s.tloop.k_max = 10000
    s.tline.step = 0.05
    s.tloop.verbose = True
    s.run()
    print('F', np.max(
        np.sum(s.hist.F_t[:, s.right_x_s.dofs]), axis=-1)
    )
    w = s.get_window()
    w.configure_traits()

    ax = p.subplot(111)
    w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

#     s.geometry.L_x = 10
#     s.run()
#     w = s.get_window()
#     print('F', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))
#     ax = p.subplot(111)
#     w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

    s.f_lateral = -100
    s.run()
    w = s.get_window()
    print('F', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))
    ax = p.subplot(111)
    w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

    p.show()
