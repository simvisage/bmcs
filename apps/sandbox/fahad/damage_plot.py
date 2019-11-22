'''
Created on 16.09.2019

@author: fseemab
'''
import time

from ibvpy.fets import FETS2D4Q
from ibvpy.mats.viz2d_field import \
    Vis2DField, Viz2DField
from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DTensorField, Viz3DTensorField
from simulator.demo.viz2d_fw import Viz2DFW, Vis2DFW
from simulator.xdomain.xdomain_fe_grid import XDomainFEGrid
from view.window.bmcs_window import BMCSWindow

from apps.verify.bond_cum_damage.pullout_2d_model.pullout2d_model import PullOut2D
from apps.verify.bond_cum_damage.pullout_2d_model.verify02_quasi_pullout import verify02_quasi_pullout
import matplotlib.pyplot as plt
import numpy as np
import pylab as p
import traits.api as tr
import traitsui.api as ui


ds = 16
r_steel = ds / 2
L_x = ds * 5
r_concrete = 75
n_x = 2
n_y = 2
ax = p.subplot(111)

f_list = [0]  # [0, -5, -10, -15, -20]

for f_lateral in f_list:  # [0, -100]

    print('lateral confining pressure', f_lateral)

    s = verify02_quasi_pullout(f_lateral=f_lateral)
    s.xd_steel.trait_set(coord_min=(0, 0),
                         coord_max=(L_x, r_steel),
                         shape=(n_x, 1)
                         )
    s.xd_concrete.trait_set(coord_min=(0, r_steel),
                            coord_max=(r_steel, r_concrete),
                            shape=(n_x, n_y)
                            )
    s.u_max = 0.5
    s.tline.step = 0.01

    s.m_steel.trait_set(E=200000, nu=0.3)
    s.m_concrete.trait_set(E=29800, nu=0.3)
    s.m_ifc.trait_set(E_T=12900,
                      E_N=1e9,
                      tau_bar=4.2,  # 4.0,
                      K=0, gamma=10,  # 10,
                      c=1, S=0.0025, r=1,
                      m=0,
                      algorithmic=False)
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
w = s.get_window()
w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

s.run()
p.show()
