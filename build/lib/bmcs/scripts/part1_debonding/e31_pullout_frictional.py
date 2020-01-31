'''
Created on Jul 12, 2017

@author: rch
'''
from os.path import join

from view.window import BMCSWindow

from bmcs.pullout.pullout_analytical_sim import PullOutModel
import pylab as p


class PullOutConstantBondStudy(BMCSWindow):

    title = 'Pullout with frictional bond'
    desc = '''Analytic solution of the bond with constant shear
    bond law is demonstrated showing the shear flow, slip and strain
    along the fiber at a marked state.
'''

    def __init__(self, *args, **kw):
        self.model = PullOutModel(name='e31_pullout_frictional',
                                  n_x_e=200, w_max=1.5)
        self.model.geometry.set(L_x=800)
    #     self.cross_section.set(A_f=4.5, P_b=1.0)
    #     self.material.set(E_f=9 * 180000.0, tau_pi_bar=2.77 * 9)
        self.model.tline.step = 0.01

        # self.add_viz2d('load function')
        self.add_viz2d('F-w', 'force-displacement')
        self.add_viz2d('field', 'shear flow', plot_fn='sf')
        self.add_viz2d('field', 'displacement', plot_fn='u')
        self.add_viz2d('field', 'strain', plot_fn='eps')
    #    self.add_viz2d('field', 'sig', plot_fn='sig')

        self.model.run()

    def plot_tex(self):
        p.figure(figsize=(9, 5))
        vd = self.viz_sheet.viz2d_dict
        ax = p.subplot(221)
        vd['force-displacement'].plot_tex(ax, 1.0)
        for vot in [0.25, 0.5, 0.75, 1.0]:
            ax = p.subplot(221)
            vd['force-displacement'].plot_marker(ax, vot)
            ax = p.subplot(222)
            vd['shear flow'].plot_tex(ax, vot, label='')
            ax = p.subplot(223)
            vd['displacement'].plot_tex(ax, vot, label='')
            ax = p.subplot(224)
            vd['strain'].plot_tex(ax, vot, label='')

    def write_tex_output(self, subfile, rdir,
                         rel_study_path, itags):
        fname = 'fig_frictional_bond.pdf'
        self.plot_tex()
        p.tight_layout()
        p.savefig(join(rdir, fname))
        subfile.write(r'''
    \includegraphics[width=0.95\textwidth]{%s}
    ''' % join(rel_study_path, fname))


def run_pullout_const_shear(*args, **kw):
    w = PullOutConstantBondStudy()
    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_pullout_const_shear()
