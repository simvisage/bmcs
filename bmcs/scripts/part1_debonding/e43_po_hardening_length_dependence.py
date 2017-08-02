'''
Created on Jul 12, 2017

@author: rch
'''
from os.path import join

from bmcs.pullout.pullout_dp import PullOutModel
from reporter import ReportStudy, Reporter
from traits.api import Instance, Array, Float, List, Str, \
    Property, cached_property
import numpy as np
import pylab as p


def get_pullout_model_carbon_concrete(u_f0_max=5.0):
    '''Helper method to get the constructing the default
    configuration of the pullout model.
    '''
    '''Helper method to get the constructing the default
    configuration of the pullout model.
    '''
    po = PullOutModel(n_e_x=200, k_max=500, u_f0_max=u_f0_max)
    po.tline.step = 0.005
    po.loading_scenario.set(loading_type='cyclic',
                            amplitude_type='constant',
                            loading_range='non-symmetric'
                            )
    po.loading_scenario.set(number_of_cycles=1,
                            unloading_ratio=0.98,
                            )
    po.geometry.L_x = 100.0
    po.cross_section.set(A_f=16.67, P_b=9.0, A_m=1540.0)
    po.mats_eval.set(E_m=28480, E_f=170000)
    po.mats_eval.set(gamma=1.5, K=0.0, tau_bar=5.0)
    po.mats_eval.omega_fn.set(alpha_1=1.0, alpha_2=1, plot_max=2.8)
    return po


class PSLengthDependenceStudy(ReportStudy):

    name = Str('e43_po_hardening_length_dependence')

    title = Str(r'''Effect of embedded length on the pull-out response with hardening bond behavior
    ''')
    desc = Str(r'''This study shows the application of the bond-slip law
    with combined hardening and damage function for the simulation of
    double sided pull-out test. 
    ''')
    u_f0_max = Float(5.0)
    po = Instance(PullOutModel, report=True)

    def _po_default(self):
        u_f0_max = self.u_f0_max
        po = get_pullout_model_carbon_concrete(u_f0_max)
        po.loading_scenario.loading_type = 'monotonic'
        po.tline.step = 0.005
        return po

    lengths = Array(Float)

    def _lengths_default(self):
        return [100, 150, 200, 250, 300, 350]

    P_u_record = Property(List, depends_on='lengths')

    @cached_property
    def _get_P_u_record(self):
        po = self.po
        P_u_record = []
        for L in self.lengths:
            po.geometry.L_x = L
            po.run()
            P = po.get_P_t()
            w0, wL = po.get_w_t()
            P_u_record.append((L, P, wL))
        return P_u_record

    def plot_output(self):

        p.figure(figsize=(8, 3.5))
        p.subplot(1, 2, 1)
        po = self.po
        u_f0_max = po.u_f0_max
        A_f = po.cross_section.A_f
        sig_f_max = 1600.00
        P_f_max = A_f * sig_f_max

        p.plot([0.0, u_f0_max], [P_f_max, P_f_max], '-', label='yarn failure')

        max_P_list = []
        for L, P, u in self.P_u_record:
            p.plot(u, P, label='L=%d [mm]' % L)
            max_P_list.append(np.max(P))
        p.legend(loc=2)

        # plot the pullout force / length dependence
        p.subplot(1, 2, 2)
        p.plot([0.0, np.max(self.lengths)],
               [P_f_max, P_f_max], label='yarn failure')
        p.plot(self.lengths, max_P_list, 'o-')
        p.xlim(xmin=0)
        p.ylim(ymin=0)
        p.legend(loc=2)

    def write_tex_input(self, subfile, rdir,
                        rel_study_path, itags):
        self.po.write_tex_table(subfile, rdir,
                                rel_study_path, itags)

    def write_tex_output(self, subfile, rdir,
                         rel_study_path, itags):
        fname = 'fig_length_dependency.pdf'
        self.plot_output()
        p.tight_layout()
        p.savefig(join(rdir, fname))
        subfile.write(r'''
    \includegraphics[width=0.95\textwidth]{%s}
    ''' % join(rel_study_path, fname))


if __name__ == "__main__":
    e43 = PSLengthDependenceStudy()
    r = Reporter(studies=[e43])
    r.write()
    r.run_pdflatex()
    r.show_pdf()
