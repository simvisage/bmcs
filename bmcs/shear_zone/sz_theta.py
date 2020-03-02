'''
Created on Feb 27, 2020

@author: rch
'''

import sympy as sp

tau_fps, sigma_x, sigma_z = sp.symbols(r'\tau_\mathrm{fps}, sigma_x, sigma_z')
f_ct = sp.Symbol('f_{\mathrm{ct}}', nonnegative=True)

sigma_xz = sp.Matrix([[sigma_x, tau_fps],
                      [tau_fps, sigma_z]])
sigma_x0 = sigma_xz.subs(sigma_z, 0)

P_xz, D_xz = sigma_xz.diagonalize()
P_x0, D_x0 = P_xz.subs(sigma_z, 0), D_xz.subs(sigma_z, 0)

subs_sigma_z = sp.solve({D_xz[1, 1] - f_ct}, {sigma_z})[0]
P_xf = P_xz.subs(subs_sigma_z)

theta_f = sp.atan(sp.simplify(-P_xf[0, 0] / P_xf[1, 0]))
theta_0 = sp.atan(sp.simplify(-P_x0[0, 0] / P_x0[1, 0]))

get_theta_f = sp.lambdify((tau_fps, sigma_x, f_ct), theta_f)
get_theta_0 = sp.lambdify((tau_fps, sigma_x, f_ct), theta_0)