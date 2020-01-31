'''
Created on Jul 12, 2017

@author: rch
'''
from bmcs.pullout.pullout_analytical_model import PullOutModel
from view.window import BMCSWindow


def construct_pullout_study():
    po = PullOutModel(name='t31_pullout_frictional', n_x=200, w_max=1.5)
    po.title = 'Pullout with frictional bond'
    po.desc = '''Analytic solution of the bond with constant shear
    bond law is demonstrated showing the shear flow, slip and strain
    along the fiber at a marked state.
'''

    po.geometry.set(L_x=800)
#     po.cross_section.set(A_f=4.5, P_b=1.0)
#     po.material.set(E_f=9 * 180000.0, tau_pi_bar=2.77 * 9)
    po.tline.step = 0.01

    po.run()
    w = BMCSWindow(model=po)
    # po.add_viz2d('load function')
    po.add_viz2d('F-w')
    po.add_viz2d('field', 'shear flow', plot_fn='sf')
    po.add_viz2d('field', 'displacement', plot_fn='u')
    po.add_viz2d('field', 'strain', plot_fn='eps')
#    po.add_viz2d('field', 'sig', plot_fn='sig')

    return w


def run_pullout_const_shear(*args, **kw):
    w = construct_pullout_study()
    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_pullout_const_shear()
