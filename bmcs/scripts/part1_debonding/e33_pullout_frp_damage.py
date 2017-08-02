'''
Created on Jul 12, 2017

@author: rch
'''
from os.path import join

from bmcs.pullout.pullout_frp_damage import PullOutModel
from view.window import BMCSWindow

import pylab as p


class PullOutFRPDamageBondStudy(BMCSWindow):

    title = 'Pullout with bond damage-softening'
    desc = '''This example demonstrates the local nature of debonding
    with propagating damage process zone. Notice the constant 
    energy release rate.
'''

    def __init__(self, *args, **kw):
        self.model = PullOutModel(name='e33_pullout_frp_damage',
                                  n_e_x=100, k_max=500, u_f0_max=0.3)
        self.model.tline.step = 0.01
        self.model.geometry.L_x = 500.0
        self.model.loading_scenario.set(loading_type='monotonic')
        self.model.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
        # self.model.mats_eval.set()
        self.model.mats_eval.omega_fn.set(b=10.4, Gf=1.19, plot_max=0.5)
        self.model.run()

        self.add_viz2d('load function', 'load-time')
        self.add_viz2d('F-w', 'load-displacement')
        self.add_viz2d('field', 'u_C', plot_fn='u_C')
        self.add_viz2d('field', 'omega', plot_fn='omega')
        self.add_viz2d('field', 'eps_C', plot_fn='eps_C')
        self.add_viz2d('field', 's', plot_fn='s')
        self.add_viz2d('field', 'sig_C', plot_fn='sig_C')
        self.add_viz2d('field', 'sf', plot_fn='sf')
        self.add_viz2d('dissipation', 'dissipation')
        self.add_viz2d('dissipation rate', 'dissipation rate')

    def plot_fig1_tex(self):
        p.figure(figsize=(8, 4))
        vd = self.viz_sheet.viz2d_dict
        ax = p.subplot(221)
        vd['load-time'].plot_tex(ax, 1.0)
        ax = p.subplot(222)
        vd['load-displacement'].plot_tex(ax, 1.0)
        for vot in [0.25, 0.5, 0.75, 1.0]:
            ax = p.subplot(221)
            vd['load-time'].plot_marker(ax, vot)
            ax = p.subplot(222)
            vd['load-displacement'].plot_marker(ax, vot)
            ax = p.subplot(223)
            vd['u_C'].plot_tex(ax, vot, label_m='', label_f='')
            ax = p.subplot(224)
            vd['eps_C'].plot_tex(ax, vot, label_m='', label_f='')

    def plot_fig2_tex(self):
        p.figure(figsize=(8, 4))
        vd = self.viz_sheet.viz2d_dict
        for vot in [0.25, 0.5, 0.75, 1.0]:
            ax = p.subplot(221)
            vd['s'].plot_tex(ax, vot)
            ax = p.subplot(222)
            vd['sig_C'].plot_tex(ax, vot)
            ax = p.subplot(223)
            vd['sf'].plot_tex(ax, vot)
        ax = p.subplot(224)
        vd['dissipation'].plot_tex(ax, 1.0)

    def write_tex_output(self, subfile, rdir,
                         rel_study_path, itags):
        subfile.write(r'''
    Loading is applied at the right end of the specimen. The specimen
    is fixed at the unloaded end by prescribing zero matrix displacement. 
    Four stages of loading are displayed as marked in the
    loading scenario and in the pull-out curve.
    \begin{longtable}{c}
''')
        fname = 'fig_frictional_bond01.pdf'
        self.plot_fig1_tex()
        p.tight_layout()
        p.savefig(join(rdir, fname))
        subfile.write(r'''\mbox{
    \includegraphics[width=0.8\textwidth]{%s}
    }\\
    ''' % join(rel_study_path, fname))

        fname = 'fig_frictional_bond02.pdf'
        self.plot_fig2_tex()
        p.tight_layout()
        p.savefig(join(rdir, fname))
        subfile.write(r'''
    \mbox{\includegraphics[width=0.8\textwidth]{%s}}

    ''' % join(rel_study_path, fname))
        subfile.write(r'''
    \end{longtable}
    ''')


def run_pullout_frp_damage(*args, **kw):
    w = PullOutFRPDamageBondStudy()
    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_pullout_frp_damage()
