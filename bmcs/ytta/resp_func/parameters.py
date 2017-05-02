'''
Created on Feb 9, 2010

@author: rostislav
'''

from numpy import pi
from traits.api import \
    HasTraits, Float, Range, Enum, on_trait_change, Bool, Property, Array

from traitsui.api import TextEditor


float_editor = TextEditor(
    evaluate=float, format_str="%.9g", enter_set=True, auto_set=False)


class Geometry(HasTraits):
    ''' geometrical parameters '''
    rf = Float(0.0001, auto_set=False, enter_set=True,
               label='rf', modified=True,
               desc='fiber radius')

    phi = Range(0., pi / 2., auto_set=False, enter_set=True,
                label='phi', modified=True,
                desc='angle of inclination to crack plane normal')

    l = Float(0.001, auto_set=False, enter_set=True,
              label='l[m]', modified=True,
              desc='free length')

    z = Float(0.0, enter_set=True, auto_set=False,
              label='z[m]', modified=True,
              desc='centroidal distance from crack plane')

    Lf = Float(0.01, enter_set=True, auto_set=False,
               label='lf[m]', modified=True,
               desc='fibre total length')

    L = Float(0.003, enter_set=True, auto_set=False,
              label='Le[m]', modified=True,
              desc='embedded length')

    Af = Property(Float, depends_on='rf', label='Af[m2]')

    def _get_Af(self):
        return self.rf ** 2 * pi

    p = Property(Float, depends_on='rf', label='p[m]')

    def _get_p(self):
        return self.rf * 2 * pi

    lambd = Float(0.02, enter_set=True, auto_set=False,
                  label='lambda', modified=True,
                  desc='ratio of extra (stretched) filament'
                  'length to the nominal length')

    theta = Float(0.01, enter_set=True, auto_set=False,
                  label='theta', modified=True,
                  desc='filament activation strain - slack')


class Material(HasTraits):
    ''' material parameters '''

    def __init__(self, **kw):
        super(Material, self).__init__(**kw)
        self.get_params()

    Ef = Float(auto_set=False, enter_set=True,
               label='Ef', modified=True, editor=float_editor,
               desc='fiber tensional stiffness')

    fu = Float(auto_set=False, enter_set=True,
               label='fu', modified=True,
               desc='fibre breaking stress')

    k = Float(auto_set=False, enter_set=True,
              label='k', modified=True,
              desc='matrix shear stiffness')

    qf = Array(auto_set=False, enter_set=True,
               label='qf', modified=True,
               desc='frictional stress in the debonded zone per unit length')

    G = Float(auto_set=False, enter_set=True,
              modified=True, label='Gd',
              desc='unit energy needed for the interface crack to propagate')

    qy = Array(auto_set=False, enter_set=True,
               label='qy', modified=True,
               desc='shear stress for debonding')

    beta = Float(auto_set=False, enter_set=True,
                 label='beta', modified=True,
                 desc='slip hardening coefficient')

    f = Range(0.5, 0.9, auto_set=False, enter_set=True,
              label='f', modified=True,
              desc='snubbing coefficient')

    include_fu = Bool(False, label='include fu', modified=True,
                      desc='include breaking stress')

    material_choice = Enum('glass-concrete', 'steel-concrete', 'PVA-epoxy',
                           modified=True, desc='chooses a material combination')

    material_params = {'glass-concrete': [70e9, 125e7, 21e9, [1220], 6.1, [12.1e3],
                                          0.05, 0.5],
                       'steel-concrete': [210e9, 500e9, 20e9, 1230, 6.2, 12.2e3,
                                          0.05, 0.5],
                       'PVA-epoxy': [60e9, 1660e9, 10e6, 1240, 6.3, 12.3e3,
                                     0.05, 0.5]
                       }

    def remove_listeners(self):
        pass

    def get_value(self):
        pass

    @on_trait_change('material_choice')
    def get_params(self):
        self.remove_listeners()
        mat_par = self.material_params
        choice = self.material_choice
        self.Ef = mat_par[choice][0]
        self.fu = mat_par[choice][1]
        self.k = mat_par[choice][2]
        self.qf = mat_par[choice][3]
        self.G = mat_par[choice][4]
        self.qy = mat_par[choice][5]
        self.beta = mat_par[choice][6]
        self.f = mat_par[choice][7]
        self.get_value()


class Plot(HasTraits):
    ''' parameters for plotting '''
    u_plot = Float(4e-5, auto_set=False, enter_set=True,
                   label='u[m]', modified=True,
                   desc='max plotted displacement')

    w_plot = Float(0.004, auto_set=False, enter_set=True,
                   label='w[m]', modified=True,
                   desc='max plotted crack opening')

    yvalues = Enum('forces', 'stresses', modified=True,
                   label='yaxis', desc='whether to show stresses or forces on the y-axis')


if __name__ == "__main__":
    geom = Material()
    geom.configure_traits()
