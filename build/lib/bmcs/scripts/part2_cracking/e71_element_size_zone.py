from os.path import join

from reporter import Reporter
from traits.api import \
    Float, List, Property, cached_property, Array, Instance, Int
from view.window import BMCSModel
from view.window.bmcs_window import BMCSWindow

import numpy as np
import pylab as p


class LocalizationElementModel(BMCSModel):

    E = Float(20000.0, MAT=True,
              auto_set=False, enter_set=True,
              symbol='$E$', unit='MPa', desc='E-modulus')

    f_t = Float(2.4, MAT=True,
                auto_set=False, enter_set=True,
                symbol='$f_t$', unit='MPa', desc='tensile strength')

    G_f = Float(0.09, MAT=True,
                auto_set=False, enter_set=True,
                symbol='$G_\mathrm{F}$', unit='N/mm', desc='fracture energy')

    L = Float(100.0, GEO=True,
              auto_set=False, enter_set=True,
              symbol='$L$', unit='mm', desc='length')

    n_E = Int(30, MESH=True,
              auto_set=False, enter_set=True,
              symbol='$L$', unit='mm', desc='length')

    eps_0 = Property(depends_on='+MAT')

    @cached_property
    def _get_eps_0(self):
        return self.f_t / self.E

    eps_f = Property(depends_on='+MAT')

    @cached_property
    def _get_eps_f(self):
        return self.G_f / self.f_t

    eps_max = Property(depends_on='+MAT')

    @cached_property
    def _get_eps_max(self):
        return 1.0 * self.eps_f

    # model with exponential softening law
    n_T = 100   # number of increments

    ####

    def f(self, w, f_t, G_f):
        '''Softening law'''
        return f_t * np.exp(-f_t / G_f * w)

    def F(self, w, f_t, G_f):
        '''Integral of the softening law'''
        return G_f - G_f * np.exp(-f_t / G_f * w)


class LocalizationElementStudy(BMCSWindow):

    title = 'Localization in an element within a discretization'
    desc = '''Given a discretization of a tensile test with $n$ elements 
    and one element with localized softening, let us study  
'''

    colors = List(['orange', 'red', 'green', 'blue', 'gray', 'yellow'])

    def __init__(self, *args, **kw):
        self.model = LocalizationElementModel(name='e71_element_localization')

    n_E_list = List([], GEO=True,
                    auto_set=False, enter_set=True,
                    symbol='$n_\mathrm{E}$', unit='-', desc='number of elements')

    def _n_E_list_default(self):
        return [10, 20, 30]

    def plot_study_for_n_E(self):
        p.figure(figsize=(9, 4))
        p.subplot(121)
        n_E_list = self.n_E_list
        for n_e in n_E_list:  # n: number of elements
            sig_t = [0]
            eps_t = [0]
            L_s = 1. / n_e
            for sig in np.linspace(self.model.f_t, 0.1, self.model.n_T):
                f_t = self.model.f_t
                G_f = self.model.G_f
                L = self.model.L
                E = self.model.E
                eps_s = -f_t / G_f * np.log(sig / f_t)
                u_e = sig / E * (L - L_s)
                u_s = L_s * eps_s
                u = u_e + u_s
                sig_t.append(sig)  # store the values of stress level
                eps_t.append(u / L)  # store the values of the average strain
            p.plot(eps_t, sig_t, label='n=%i' % n_e)
            p.legend(loc=1)

        p.xlabel('strain')
        p.ylabel('stress')

        p.subplot(122)

        # model with linear softening law
        for n_e in n_E_list:  # n: number of elements
            eps_f = G_f / f_t
            eps = np.array([0.0, f_t / E, eps_f / n_e])
            sig = np.array([0.0, f_t, 0.0])
            p.plot(eps, sig, label='n=%i' % n_e)
            p.legend(loc=1)

        p.xlabel('strain')
        p.ylabel('stress')

    def write_tex_input(self, subfile, rdir,
                        rel_study_path, itags):
        self.model.write_tex_table(subfile, rdir,
                                   rel_study_path, itags)

    def write_tex_output(self, subfile, rdir,
                         rel_study_path, itags):
        fname = 'fig_element_length_dependence.pdf'
        subfile.write('''Response for varied number of elements $n_\mathrm{E} \in %s$
        ''' % self.n_E_list)
        self.plot_study_for_n_E()
        p.tight_layout()
        p.savefig(join(rdir, fname))
        subfile.write(r'''\includegraphics[width=0.95\textwidth]{%s}
    ''' % join(rel_study_path, fname))


if __name__ == "__main__":
    lz = LocalizationElementStudy()
    r = Reporter(studies=[lz])
    r.write()
    r.show_tex()
    r.run_pdflatex()
    r.show_pdf()
