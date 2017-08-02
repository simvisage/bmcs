from os.path import join

from reporter import Reporter
from traits.api import \
    Float, List, Property, cached_property, Array, Instance
from view.window import BMCSModel
from view.window.bmcs_window import BMCSWindow

import numpy as np
import pylab as p


class LocalizationZoneModel(BMCSModel):
    f_t = Float(2.5, MAT=True,
                auto_set=False, enter_set=True,
                symbol='$f_t$', unit='MPa', desc='tensile strength')
    E = Float(34000.0, MAT=True,
              auto_set=False, enter_set=True,
              symbol='$E$', unit='MPa', desc='E-modulus')
    L = Float(300.0, GEO=True,
              auto_set=False, enter_set=True,
              symbol='$L$', unit='mm', desc='length')
    G_f = Float(0.014, MAT=True,
                auto_set=False, enter_set=True,
                symbol='$G_\mathrm{F}$', unit='N/mm', desc='fracture energy')
    A = Float(10.0, CS=True,
              auto_set=False, enter_set=True,
              symbol='$A$', unit='$\mathrm{mm}^2$', desc='cross-sectional area')

    def f(self, w, f_t, G_f):
        '''Softening law'''
        return f_t * np.exp(-f_t / G_f * w)

    def F(self, w, f_t, G_f):
        '''Integral of the softening law'''
        return G_f - G_f * np.exp(-f_t / G_f * w)

    w_ch = Property(Float(depends_on='+MAT'))

    def _get_w_ch(self):
        return self.G_f / self.f_t

    w = Property(Array(np.float, depends_on='+MAT'))

    @cached_property
    def _get_w(self):
        w_max = 5.0 * self.w_ch
        return np.linspace(0, w_max, 100)

    def get_response(self):
        E = self.E
        A = self.A
        G_f = self.G_f
        f_t = self.f_t
        L = self.L
        w = self.w
        eps_el = [0, f_t / E]
        sig_el = [0, f_t]
        eps_w = 1 / E * self.f(w, f_t, G_f) + w / L
        sig_w = self.f(w, f_t, G_f)
        W_el = [0, f_t**2 / 2 / E * A * L]
        U_el = [0, f_t**2 / 2 / E * A * L]
        W_w = 1. / 2. / E * A * L * \
            self.f(w, f_t, G_f)**2 + A * self.F(w, f_t, G_f)
        U_w = 1. / 2. / E * A * L * \
            self.f(w, f_t, G_f)**2 + 1. / 2. * \
            A * self.f(w, f_t, G_f) * w
        eps = np.hstack([eps_el, eps_w])
        sig = np.hstack([sig_el, sig_w])
        W = np.hstack([W_el, W_w])
        U = np.hstack([U_el, U_w])

        return eps, sig, W, U


class LocalizationZoneStudy(BMCSWindow):

    title = 'Tensile response of a bar with crack localization'
    desc = '''This example demonstrates stress-strain response 
    of a bar with a single cross section exhibiting softening upon
    reaching the strength $f_t$.
'''

    colors = List(['orange', 'red', 'green', 'blue', 'gray', 'yellow'])

    def __init__(self, *args, **kw):
        self.model = LocalizationZoneModel(name='e51_length_dependence')

    L_el_list = List([100.0, 200.0, 300.0, 500.0, 1000.0], GEO=True,
                     auto_set=False, enter_set=True,
                     symbol='$L_\mathrm{el}$', unit='mm', desc='length values')
    L_arr = Property(depends_on='L_el_list, L_el_list_items')

    @cached_property
    def _get_L_arr(self):
        return np.array(self.L_el_list)

    Gf_list = List([0.010, 0.040, 0.070], MAT=True,
                   auto_set=False, enter_set=True,
                   symbol='$G_\mathrm{F}$', unit='Nmm', desc='fracture energy values')
    Gf_arr = Property(depends_on='Gf_list, Gf_list_items')

    @cached_property
    def _get_Gf_arr(self):
        return np.array(self.Gf_list)
    records = List([])

    def run_study_for_L(self):
        self.records = []
        for L in self.L_arr:
            self.model.L = L
            self.records.append(self.model.get_response())

    def run_study_for_Gf(self):
        self.model.L = 100.0
        self.records = []
        for Gf in self.Gf_arr:
            self.model.G_f = Gf
            self.records.append(self.model.get_response())

    def plot_output(self, P_arr, var='L'):
        p.figure(figsize=(9.0, 6.0))
        w = self.model.w
        alpha = 0.1
        lw = 2
        for P, record, c in zip(P_arr, self.records, self.colors):
            setattr(self.model, var, P)
            eps, sig, W, U = record
            G_f = self.model.G_f
            f_t = self.model.f_t
            E = self.model.E
            p.subplot(2, 2, 1)
            p.plot(eps, sig, lw=lw, color=c, label='%s=%g' % (var, P))
            p.xlabel('strain [-]')
            p.ylabel('stress [MPa]')
            p.legend(loc=1)
            p.fill_between(eps, 0, sig, facecolor=c, alpha=alpha)
            p.subplot(2, 2, 2)
            p.plot(w, self.model.f(w, f_t, G_f), color=c)
            p.xlabel('crack opening [mm]')
            p.ylabel('stress [MPa]')
            p.fill_between(w, 0, self.model.f(
                w, f_t, G_f), facecolor=c, alpha=alpha)
            p.plot([0, self.model.w_ch], [f_t, 0], lw=1, color=c)
            p.subplot(2, 2, 3)
            p.plot(eps, W, lw=lw, color=c)
            p.plot(eps, U, lw=lw, color=c)
            p.fill_between(eps, U, W, facecolor=c, alpha=alpha)
            p.xlabel('strain [-]')
            p.ylabel('energy [Nmm]')
            p.subplot(2, 2, 4)
            p.plot(eps, W - U, lw=lw, color=c)
            p.fill_between(eps, W - U, facecolor=c, alpha=alpha)
            p.xlabel('strain [-]')
            p.ylabel('released energy [Nmm]')
            p.tight_layout()

    def write_tex_input(self, subfile, rdir,
                        rel_study_path, itags):
        self.model.write_tex_table(subfile, rdir,
                                   rel_study_path, itags)

    def write_tex_output(self, subfile, rdir,
                         rel_study_path, itags):
        fname = 'fig_length_dependence.pdf'
        subfile.write(r'''\begin{center}
''')
        subfile.write(r'''Response for varied length $L \in %s$\\
        ''' % self.L_el_list)
        self.run_study_for_L()
        self.plot_output(self.L_arr, 'L')
        p.tight_layout()
        p.savefig(join(rdir, fname))
        subfile.write(r'''\includegraphics[width=0.95\textwidth]{%s}\\
    ''' % join(rel_study_path, fname))

        fname = 'fig_fracture_energy_dependence.pdf'
        subfile.write(r'''Response for varied fracture energy $G_\mathrm{F} \in %s$\\
        ''' % self.Gf_list)
        self.run_study_for_Gf()
        self.plot_output(self.Gf_arr, 'G_f')
        p.tight_layout()
        p.savefig(join(rdir, fname))
        subfile.write(r'''\includegraphics[width=0.95\textwidth]{%s}\\
    ''' % join(rel_study_path, fname))
        subfile.write(r'''\end{center}
''')


if __name__ == "__main__":
    lz = LocalizationZoneStudy()
    r = Reporter(studies=[lz])
    r.write()
    r.show_tex()
    r.run_pdflatex()
    r.show_pdf()
