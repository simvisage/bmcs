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
    po = PullOutModel(n_e_x=100, k_max=1000, u_f0_max=u_f0_max)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.geometry.L_x = 100.0
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.material.set(K=100000.0, gamma=0.0, tau_bar=45.0)
    po.material.omega_fn.set(alpha_1=1.0, alpha_2=50.0, plot_max=0.62)

    return po


class PSLengthDependenceStudy(ReportStudy):

    name = Str('e44_po_softening_length_dependence')

    title = Str(r'''Effect of embedded length on the pull-out response with softening bond
    ''')
    desc = Str(r'''This study shows the application of the bond-slip law
    with softening bond behavior for the simulation of a pull-out test. 
    ''')
    u_f0_max = Float(3.0)
    po = Instance(PullOutModel, report=True)

    def _po_default(self):
        u_f0_max = self.u_f0_max
        po = get_pullout_model_carbon_concrete(u_f0_max)
        po.loading_scenario.loading_type = 'monotonic'
        po.tline.step = 0.005
        return po

    lengths = Array(Float)

    def _lengths_default(self):
        L_max = np.log10(350.0)
        L_min = np.log10(10.0)
        return np.logspace(L_min, L_max, 7)

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

    def plot(self):

        p.figure(figsize=(8, 3))
        p.subplot(1, 2, 1)
        po = self.po
        u_f0_max = po.u_f0_max

        sig_f_max = 2400.00
        A_f = po.cross_section.A_f
        P_f_max = A_f * sig_f_max

        p.plot([0.0, u_f0_max], [P_f_max, P_f_max], '-', label='yarn failure')

        max_P_list = []
        for L, P, u in self.P_u_record:
            p.plot(u, P, label='L=%d [mm]' % L)
            max_P_list.append(np.max(P))
        p.legend(loc=1)
        p.xlim(xmin=0)
        p.ylim(ymin=0)

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
        self.plot()
        p.tight_layout()
        fname = 'fig_length_dependence.pdf'
        subfile.write(r'''
    \includegraphics[width=0.95\textwidth]{%s}
    ''' % join(rel_study_path, fname))
        p.savefig(join(rdir, fname))


if __name__ == "__main__":

    e43 = PSLengthDependenceStudy()
    r = Reporter(studies=[e43])
    r.write()
    # r.show_tex()
    r.run_pdflatex()
    r.show_pdf()
