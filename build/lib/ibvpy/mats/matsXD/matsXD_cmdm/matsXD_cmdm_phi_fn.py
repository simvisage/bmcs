
from math import exp, pow
from numpy import \
    array, ones, frompyfunc, \
    sqrt, linspace, where, trapz, \
    hstack
from traits.api import \
    Float,  \
    Instance, HasStrictTraits, on_trait_change,  \
    provides, Button, \
    Interface, WeakRef
from traitsui.api import \
    Item, View, VGroup, \
    Group, HGroup
from mathkit.mfn import MFnLineArray
from mathkit.mfn.mfn_line.mfn_matplotlib_editor import MFnMatplotlibEditor
from mathkit.mfn.mfn_line.mfn_plot_adapter import MFnPlotAdapter

mfn_editor = MFnMatplotlibEditor(
    adapter=MFnPlotAdapter(label_x='strain',
                                   label_y='integrity',
                                   title='Softening law for a microplane',
                                   # Plot properties
                                   line_color=["black"],
                                   bgcolor="white",
                                   max_size=(360, 260),
                                   # Border, padding properties
                                   border_visible=False,
                           padding={'left': 0.15,
                                    'right': 0.9,
                                    'bottom': 0.15,
                                    'top': 0.85},
                           padding_bg_color="white"
                           ))


class IPhiFn(Interface):

    '''Interface to adamage function representation
    '''

    def identify_parameters(self):
        '''Return the set of parameters defining the respective damage function.
        '''

    def __call__(self, e_max, *c_list):
        '''return the value of the damage function for the,
        maximum strain achieved so far and list of coefficients for material parameters
        '''

#-------------------------------------------------------------------------
# Damage function for MDM
#-------------------------------------------------------------------------


@provides(IPhiFn)
class PhiFnBase(HasStrictTraits):
    '''
    Damage function.
    '''
    polar_discr = WeakRef(transient=True)

    def __init__(self, **args):
        super(PhiFnBase, self).__init__(**args)
#        self.refresh_plot()

    mfn = Instance(MFnLineArray)

    def _mfn_default(self):
        return MFnLineArray(xdata=[0, 1], ydata=[1, 1])

    def refresh_plot(self):
        x_min, x_max = self.get_plot_range()
        x = linspace(x_min, x_max, 100)
        phi_fn = frompyfunc(self, 1, 1)
        x_ = x
        y_ = phi_fn(x)
        y = array([v for v in y_], dtype='float')
#        self.mfn.set(xdata = x, ydata = y)
#        self.mfn.data_changed = True

    def __call__(self, e_max, *c_list):
        return self.get_value(e_max, *c_list)

    def get_plot_range(self):
        '''
        Get the range of the plot domain
        '''
        raise NotImplementedError

    def identify_parameters(self):
        '''
        Extract the traits that are of type Float
        '''
        params = []
        for name, trait in list(self.traits().items()):
            if trait.is_trait_type(Float):
                params.append(name)
        return params

    def fit_params(self, *params):
        '''Possiblity to adapt the microplane-related
        material paramters based on the integral characteric specification.
        '''
        return

#-------------------------------------------------------------------------
# Piecewise Linear damage function for MDM
#-------------------------------------------------------------------------


class PhiFnGeneral(PhiFnBase):

    # implements(IPhiFn)

    def _polar_discr_changed(self):
        self.polar_discr.regularization = False

    def get_value(self, e_max, *c_list):
        '''
        Evaluate the integrity of a particular microplane.
        '''
        return self.mfn.get_value(e_max)

    def get_integ(self, e_max, *c_list):
        '''
        Evaluate the integrity of a particular microplane.
        '''
        # get the data that defines PhiFnGeneral
        # (fitted from experiment)
        _xdata = self.mfn.xdata
        _ydata = self.mfn.ydata

        # get the values smaller then the current e_max
        _xdata_ = _xdata[where(_xdata < e_max)[0]]
        _ydata_ = _ydata[:len(_xdata_)]
#        print '_xdata_' , _xdata_

        # add the value pair for e_max
        _xdata_emax = hstack([_xdata_, e_max])
        _ydata_emax = hstack([_ydata_, self.mfn.get_value(e_max)])
#        print '_xdata_emax' , _xdata_emax

        # assume an uncoupled relation (e.g. direct link) between the microplane
        # strains (e) and microplane stresses (s), e.g. s = phi * E * phi * e;
        # The methode 'get_integ' returns the value without the young's modulus,
        # which is multiplied in 'PhiFnPolar';
        # _ydata_integ = phi * phi * e
        #
        # @todo: this is only approximately true!; the correct evaluation
        # takes the version consistend (stiffness/compliance) pairs for
        # the microplane strains and stresses (work conjugates)
        _ydata_integ = _ydata_emax * _ydata_emax * _xdata_emax

        # integral under the stress-strain curve
        E_t = trapz(_ydata_integ, _xdata_emax)
        # area of the stored elastic energy
        U_t = 0.0
        if len(_xdata_emax) != 0:
            U_t = 0.5 * _ydata_integ[-1] * _xdata_emax[-1]
#        print 'E_t', E_t
#        print 'U_t', U_t
#        print 'E_t - U_t', E_t - U_t
        return E_t - U_t

    def get_plot_range(self):
        return self.mfn.xdata[0], self.mfn.xdata[-1]

    print_button = Button

    @on_trait_change('print_button')
    def print_button_fired(self):
        print('eps:\n', [self.mfn.xdata])
        print('1-omega:\n', [self.mfn.ydata])

    # Default TraitsUI view
    traits_view = View(Group(
        Item('mfn', show_label=False, editor=mfn_editor),
        Item('print_button', show_label=False),
        label='Damage law',
        show_border=True
    ),
        buttons=['OK', 'Cancel'],
        resizable=True,
        width=800, height=800)


#-------------------------------------------------------------------------
# Piecewise Linear damage function with drop to zero for MDM (used within 'MATSCalibDamageFn')
#-------------------------------------------------------------------------
class PhiFnGeneralExtended(PhiFnGeneral):

    # implements(IPhiFn)

    factor_eps_fail = Float(1.0)

    def get_value(self, e_max, *c_list):
        '''
        Evaluate the integrity of a particular microplane.
        Overload the 'get_value' method of 'PhiFnGeneral'
        and add an additional constant level and a drop
        down to zero after failure strain has been reached.
        '''
        eps_last = self.mfn.xdata[-1]
        phi_last = self.mfn.ydata[-1]
        eps_fail = eps_last * self.factor_eps_fail

        if e_max <= eps_last:
            return super(PhiFnGeneralExtended, self).get_value(e_max, *c_list)

        elif (e_max > eps_last and e_max < eps_fail):
            return phi_last
        else:
            return 1e-50

    def get_plot_range(self):
        '''plot the extended phi function'''
        return self.mfn.xdata[0], self.mfn.xdata[-1] * self.factor_eps_fail * 1.1


class PhiFnGeneralExtendedExp(PhiFnGeneral):

    # implements(IPhiFn)

    Dfp = Float(0.0, desc='residual integrity',
                enter_set=True, auto_set=False)
    Efp_frac = Float(0.01, desc='Efp factor',
                     enter_set=True, auto_set=False)

    def get_value(self, e_max, *c_list):
        '''
        Evaluate the integrity of a particular microplane.
        Overload the 'get_value' method of 'PhiFnGeneral'
        and add an exponential softening branch after
        failure strain has been reached.
        '''
        eps_last = self.mfn.xdata[-1]
        phi_last = self.mfn.ydata[-1]

        if e_max <= eps_last:
            return super(PhiFnGeneralExtendedExp, self).get_value(e_max)
        else:
            print('**** Entered softening branch ****')
            raise ValueError("Entered softening branch")
            # exponential softening with residual integrity after rupture
            # strain in the tensile test has been reached
            Dfp = self.Dfp
            Epp = eps_last
            Efp = self.Efp_frac * eps_last
            print('phi_last:', phi_last, ', eps_last:', eps_last, end=' ')
            phi = phi_last * \
                ((1 - Dfp) *
                 sqrt(Epp / e_max * exp(-(e_max - Epp) / Efp)) + Dfp)
            print(', e_max:', e_max, 'phi:', phi)
            return phi

    def get_plot_range(self):
        '''plot the extended phi function'''
        return self.mfn.xdata[0], self.mfn.xdata[-1] * 2.0

#-------------------------------------------------------------------------
# Damage function with stain softening for MDM
#-------------------------------------------------------------------------


class PhiFnStrainSoftening(PhiFnBase):

    '''
    Damage function.
    '''

    # implements(IPhiFn)

    G_f = Float(0.001117,
                label='G_f',
                desc='fracture energy',
                auto_set=False, enter_set=True)
    f_t = Float(2.8968,
                label='f_t',
                desc='tensile strength',
                auto_set=False, enter_set=True)
    md = Float(0.0,
               label='md',
               desc='factor affecting the compresive strength (explain more precisely)',
               auto_set=False, enter_set=True)
    h = Float(1.0,
              label='h',
              desc='element size to norm the fracture energy',
              auto_set=False, enter_set=True)

    Epp = Float(
        desc='strain at the onset of damage', enter_set=True, auto_set=False)
    Efp = Float(desc='strain at total damaged', enter_set=True, auto_set=False)

    @on_trait_change('G_f,f_t,md,h,polar_discr.E')
    def fit_microplane_params(self):
        '''
        Calculate the parameters of the damage function
        '''
        if self.polar_discr == None:
            return
        E = self.polar_discr.E
        G_f = self.G_f
        f_t = self.f_t
        md = self.md
        h = self.h

        gamma = (E * G_f) / (h * f_t ** 2)
        if gamma < 2.0:
            print('WARNING: elements too big -> refine, h should be at maximum only half of the characteristic length')
            print('in FIT PARAMS: gamma set to 2.0')
            gamma = 2.0

        Epp = f_t / \
            ((E * (1 - md) ** 2) * (1.95 - 0.95 / (gamma - 1) ** (0.5)))
        Efp = (G_f / ((1 - md) * h * E * Epp) +
               (2.13 - 1.13 * md) * Epp) / (2.73 - md) - Epp
        self.Epp = Epp
        self.Efp = Efp
        # @todo - plotting must be done separately
        # self.refresh_plot()

    def _polar_discr_changed(self):
        self.polar_discr.regularization = True

    def get_plot_range(self):
        return 0, self.Epp * 20.

    def get_integ(self, e_max, *c_list):
        '''
        OBSOLETE method - was used for decoupled evaluation of fracture
        energy contribution of the microplane.

        The method returns the value of the following integral:
        int( Phi(e_max~)**2 * e_max~, e_max~ = 0..e_max )
        The value corresponds to the fracture energy of the considered microplane
        divided by E. (Note: For a damage function Phi(e_max) the microplane stress-strain curve
        evaluates to s = Phi(e_max)*E*Phi(e_max)*e_max.)
        '''
        if len(c_list) == 0:
            c_list = [1., 1.]
        Epp = self.Epp * c_list[0]
        Efp = self.Efp * c_list[1]
        #
        # cf. derivation in Maple 'relation_between_Gf-ft_and_Epp-Efp
        # devide the result by E, i.e. the returned value does NOT include E
        if e_max <= Epp:
            return 0
        else:
            return -0.5 * Epp * (-Epp - 2.0 * Efp + 2.0 * Efp * exp(((-e_max + Epp) / Efp))) \
                   - 0.5 * e_max * Epp * exp(-((e_max - Epp) / Efp))

    def get_value(self, e_max, *c_list):
        '''
        Evaluate the integrity of a particular microplane.
        '''
        if len(c_list) == 0:
            c_list = [1., 1.]
        Epp = self.Epp * c_list[0]
        Efp = self.Efp * c_list[1]
        if e_max <= Epp:
            return 1.0
# @todo: check if this is necessary:
# if values smaller then 1.e-310 are returned
# a zero division error occures otherwise!
#        elif (e_max-Epp)/Efp >= 50:
#            return 1e-200
        else:
            return sqrt(Epp / e_max * exp(-(e_max - Epp) / Efp))

    # Default TraitsUI view
    traits_view = View(HGroup(
        VGroup(
            Group(
                Item('G_f'),
                Item('f_t'),
                Item('md'),
                Item('h'),
                show_border=True,
                label='Macroscopic damage parameters',
                springy=True,
            ),
            Group(Item('Epp', style='readonly'),
                  Item('Efp', style='readonly'),
                  show_border=True,
                  label='Microplane damage parameters',
                  springy=True,
                  ),
            springy=True,
        ),
        Group(
            Item('mfn', show_label=False, editor=mfn_editor),
            show_border=True,
            label='Damage function',
        ),
    ),
        buttons=['OK', 'Cancel'],
        resizable=True,
        width=800, height=500)


#-------------------------------------------------------------------------
# Damage function with residual damage level for MDM
#-------------------------------------------------------------------------
class PhiFnStrainHardeningLinear(PhiFnBase):

    '''
    Damage function which leads to a piecewise linear stress strain response.
    '''

    # implements(IPhiFn)

    E_f = Float(70e+3, desc='E-Modulus of the fibers',
                enter_set=True, auto_set=False, modified=True)
    E_m = Float(34e+3, desc='E-Modulus of the matrix',
                enter_set=True, auto_set=False, modified=True)
    rho = Float(0.03, desc='reinforcement ratio',
                enter_set=True, auto_set=False, modified=True)
    sigma_0 = Float(5.0, desc='asymptotic damage level',
                    enter_set=True, auto_set=False, modified=True)
    alpha = Float(0.0, desc='Slope of the strain hardening curve in section II',
                  enter_set=True, auto_set=False, modified=True)
    beta = Float(0.0, desc='Slope of the strain hardening curve in section III',
                 enter_set=True, auto_set=False, modified=True)
    Elimit = Float(0.006, desc='microplane strain at ultimate failure',
                   enter_set=True, auto_set=False, modified=True)

    def identify_parameters(self):
        return ['E_f', 'E_m', 'rho', 'sigma_0', 'alpha', 'beta', 'Elimit']

    def get_plot_range(self):
        return 0.0, self.Elimit * 1.2

    @on_trait_change('+modified')
    def refresh_plot(self):
        print('refreshing')
        super(PhiFnStrainHardeningLinear, self).refresh_plot()

    def _polar_discr_changed(self):
        print('regularizatoin set to False')
        self.polar_discr.regularization = False

    def get_value(self, e_max, *c_list):
        '''
        Evaluate the integrity of a particular microplane.
        '''
#        print 'x3c_list', c_list
        if len(c_list) == 0:
            c_list = [1., 1., 1., 1., 1., 1., 1.]
#        print 'x4c_list ', c_list

        E_f = self.E_f * c_list[0]
        E_m = self.E_m * c_list[1]
        rho = self.rho * c_list[2]
        sigma_0 = self.sigma_0 * c_list[3]
        alpha = self.alpha * c_list[4]
        beta = self.beta * c_list[5]
        Elimit = self.Elimit * c_list[6]
        #
        E_c = E_m * (1 - rho) + E_f * rho

        epsilon_0 = sigma_0 / E_c

        if e_max <= epsilon_0:
            return 1.0
        elif e_max >= Elimit:
            print('********** microplane reached maximum strain *********')
            return 1e-100

        epsilon_1 = sigma_0 * (-rho * E_f - E_m + rho * E_m + rho * E_f *
                               alpha + beta * E_m - beta * E_m * rho) / \
                              (rho * E_f + E_m - rho * E_m) / rho / E_f / \
                              (alpha - 1.0)

        epsilon = e_max

        if epsilon < epsilon_1:
            return sqrt(1.0 + (sigma_0 * rho * E_f + sigma_0 * E_m - sigma_0 * rho * E_m + rho * rho * E_f * E_f *
                               alpha * epsilon + rho * E_f * alpha * epsilon * E_m - rho * rho * E_f * alpha * epsilon * E_m - rho * E_f *
                               alpha * sigma_0 - epsilon * rho * rho * E_f * E_f - 2.0 * epsilon * rho * E_f * E_m + 2.0 * epsilon * rho *
                               rho * E_f * E_m - epsilon * E_m * E_m + 2.0 * epsilon * rho * E_m * E_m - epsilon * rho * rho * E_m * E_m) /
                        pow(rho * E_f + E_m - rho * E_m, 2.0) / epsilon)
        else:
            return sqrt(1.0 + E_m * (-E_f * rho * epsilon + epsilon * rho * rho * E_f + beta * sigma_0 - beta *
                                     sigma_0 * rho - epsilon * E_m + 2.0 * epsilon * rho * E_m - epsilon * rho * rho * E_m) /
                        pow(rho * E_f + E_m - rho * E_m, 2.0) / epsilon)

    # Default TraitsUI view
    traits_view = View(HGroup(
        Group(Item('E_f'),
              Item('E_m'),
              Item('rho'),
              Item('sigma_0'),
              Item('alpha'),
              Item('beta'),
              Item('Elimit'),
              springy=True,
              ),
        Item('mfn', show_label=False, editor=mfn_editor),
        label='Damage function',
        show_border=True
    ),
        buttons=['OK', 'Cancel'],
        resizable=True,
        width=800, height=400)

#-------------------------------------------------------------------------
# Damage function with residual damage level for MDM
#-------------------------------------------------------------------------


class PhiFnStrainHardening(PhiFnBase):

    '''
    Damage function.
    '''

    # implements(IPhiFn)

    Epp = Float(5.9e-05, desc='microplane strain at the onset of damage',
                enter_set=True, auto_set=False)
    Efp = Float(1.91e-04, desc='microplane strain at totaly damaged state',
                enter_set=True, auto_set=False)
    Dfp = Float(0.4, desc='asymptotic damage level',
                enter_set=True, auto_set=False)
    Elimit = Float(8.00e-02, desc='microplane strain at ultimate failure',
                   enter_set=True, auto_set=False)

    def identify_parameters(self):
        return ['Epp', 'Efp', 'Dfp', 'Elimit']

    def get_plot_range(self):
        return 0.0, self.Elimit * 1.2

    @on_trait_change('Epp,Efp,Dfp,Elimit')
    def refresh_plot(self):
        super(PhiFnStrainHardening, self).refresh_plot()

    def _polar_discr_changed(self):
        self.polar_discr.regularization = False

    def get_integ(self, e_max, *c_list):
        '''
        OBSOLETE method - was used for decoupled evaluation of fracture
        energy contribution of the microplane.

        The method returns the value of the following integral:
        int( Phi(e_max~)**2 * e_max~, e_max~ = 0..e_max )
        The value corresponds to the fracture energy of the considered microplane
        divided by E. (Note: For a damage function Phi(e_max) the microplane stress-strain curve
        evaluates to s = Phi(e_max)*E*Phi(e_max)*e_max.)
        '''
        if len(c_list) == 0:
            c_list = [1., 1.]
        Epp = self.Epp * c_list[0]
        Efp = self.Efp * c_list[1]
        Dfp = self.Dfp * c_list[2]
        Elimit = self.Elimit * c_list[3]
        # @todo: modify this for the case tension stiffening
        if e_max <= Epp:
            return 0
        else:
            return -0.5 * Epp * (-Epp - 2.0 * Efp + 2.0 * Efp * exp(((-e_max + Epp) / Efp))) \
                   - 0.5 * e_max * Epp * exp(-((e_max - Epp) / Efp))

    def get_value(self, e_max, *c_list):
        '''
        Evaluate the integrity of a particular microplane.
        '''
#        print 'x3c_list', c_list
        if len(c_list) == 0:
            c_list = [1., 1., 1., 1.]
#        print 'x4c_list ', c_list

#        print 'self.Epp used for TensionStiffening:', self.Epp
#        print 'self.Efp used for TensionStiffening:', self.Efp
#        print 'self.Dfp used for TensionStiffening:', self.Dfp
#        print 'self.Elimit used for TensionStiffening:', self.Elimit

        Epp = self.Epp * c_list[0]
        Efp = self.Efp * c_list[1]
        Dfp = self.Dfp * c_list[2]
        Elimit = self.Elimit * c_list[3]
        #
        if e_max <= Epp:
            return 1.0
        elif e_max >= Elimit:
            return 1.0e-100
# @todo: check if this is neccessary:
# if values smaller then 1.e-310 are returned
# a zero division error occures otherwise for Dfp=0!
#        elif (e_max-Epp)/Efp >= 50:
#            return Dfp
        else:
            return (1 - Dfp) * sqrt(Epp / e_max * exp(-(e_max - Epp) / Efp)) + Dfp

    # Default TraitsUI view
    traits_view = View(HGroup(
        Group(Item('Epp'),
              Item('Efp'),
              Item('Dfp'),
              Item('Elimit'),
              springy=True,
              ),
        Item('mfn', show_label=False, editor=mfn_editor),
        label='Damage function',
        show_border=True
    ),
        buttons=['OK', 'Cancel'],
        resizable=True,
        width=800, height=400)


class PhiFnStrainHardeningBezier(PhiFnBase):

    '''Fitted polynomial'''

    epsilon_0 = Float(0.00004)
    epsilon_b = Float(0.0005)
    epsilon_f = Float(0.02)
    omega_b = Float(0.63)
    omega_f = Float(0.27)
    omega_t = Float(0.27)

    def identify_parameters(self):
        return ['epsilon_0', 'epsilon_b', 'epsilon_f', 'omega_b', 'omega_f', 'omega_t']

    def _polar_discr_changed(self):
        self.polar_discr.regularization = False

    def get_plot_range(self):
        return 0.0, self.epsilon_f * 1.5

    def get_value(self, epsilon, *c_list):

        if len(c_list) == 0:
            c_list = ones((6,), dtype=float)

        epsilon_0 = self.epsilon_0 * c_list[0]
        epsilon_b = self.epsilon_b * c_list[1]
        epsilon_f = self.epsilon_f * c_list[2]
        omega_b = self.omega_b * c_list[3]
        omega_f = self.omega_f * c_list[4]
        omega_t = self.omega_t * c_list[5]

        if epsilon <= epsilon_0:
            return 1.0
        elif epsilon_0 < epsilon and epsilon <= (epsilon_0 + epsilon_b):
            return 1 - omega_b / epsilon_b * (epsilon - epsilon_0)
        elif (epsilon_0 + epsilon_b) < epsilon and epsilon <= epsilon_0 + epsilon_b + epsilon_f:
            MapleGenVar1 = (pow(1.0 - (epsilon_b * omega_t - sqrt(epsilon_b * epsilon_b *
                                                                  omega_t * omega_t - 2.0 * epsilon_b * omega_t * epsilon * omega_b + 2.0 * epsilon_b * omega_t *
                                                                  epsilon_0 * omega_b + 2.0 * epsilon_b * epsilon_b * omega_t * omega_b + epsilon_f * omega_b *
                                                                  omega_b * epsilon - epsilon_f * omega_b * omega_b * epsilon_0 - epsilon_f * omega_b * omega_b *
                                                                  epsilon_b)) / (2.0 * epsilon_b * omega_t - epsilon_f * omega_b), 2.0) * omega_b)
            MapleGenVar3 = (2.0 * (1.0 - (epsilon_b * omega_t - sqrt(epsilon_b * epsilon_b *
                                                                     omega_t * omega_t - 2.0 * epsilon_b * omega_t * epsilon * omega_b + 2.0 * epsilon_b * omega_t *
                                                                     epsilon_0 * omega_b + 2.0 * epsilon_b * epsilon_b * omega_t * omega_b + epsilon_f * omega_b *
                                                                     omega_b * epsilon - epsilon_f * omega_b * omega_b * epsilon_0 - epsilon_f * omega_b * omega_b *
                                                                     epsilon_b)) / (2.0 * epsilon_b * omega_t - epsilon_f * omega_b)) * (epsilon_b * omega_t - sqrt(
                                                                         epsilon_b * epsilon_b * omega_t * omega_t - 2.0 * epsilon_b * omega_t * epsilon * omega_b + 2.0 *
                                                                         epsilon_b * omega_t * epsilon_0 * omega_b + 2.0 * epsilon_b * epsilon_b * omega_t * omega_b +
                                                                         epsilon_f * omega_b * omega_b * epsilon - epsilon_f *
                                                                         omega_b * omega_b *
                                                                         epsilon_0 -
                                                                         epsilon_f
                                                                         * omega_b * omega_b * epsilon_b)) / (2.0 * epsilon_b * omega_t - epsilon_f * omega_b) * (omega_b
                                                                                                                                                                  + omega_t))
            MapleGenVar4 = (pow(epsilon_b * omega_t - sqrt(epsilon_b * epsilon_b * omega_t *
                                                           omega_t - 2.0 * epsilon_b * omega_t * epsilon * omega_b + 2.0 * epsilon_b * omega_t * epsilon_0 *
                                                           omega_b + 2.0 * epsilon_b * epsilon_b * omega_t * omega_b + epsilon_f * omega_b * omega_b *
                                                           epsilon - epsilon_f * omega_b * omega_b * epsilon_0 -
                                                           epsilon_f * omega_b *
                                                           omega_b * epsilon_b
                                                           ), 2.0) / pow(2.0 * epsilon_b * omega_t - epsilon_f * omega_b, 2.0) * (omega_b + omega_f))
            MapleGenVar2 = MapleGenVar3 + MapleGenVar4
            return 1 - (MapleGenVar1 + MapleGenVar2)
        elif epsilon > epsilon_0 + epsilon_b + epsilon_f:
            eps_r = epsilon_0 + epsilon_b + epsilon_f
            omega = 1 - \
                (omega_b + omega_f + omega_b / epsilon_b * (epsilon - eps_r))
            if omega <= 0.001:
                return 0.001
            else:
                return omega

    # Default TraitsUI view
    traits_view = View(Group(
        Item('mfn', show_label=False, editor=mfn_editor),
        label='Damage law',
        show_border=True
    ),
        buttons=['OK', 'Cancel'],
        resizable=True,
        width=800, height=800)


if __name__ == '__main__':
    # phi_fn = PhiFnStrainSoftening( Epp = 0.2, Efp = 0.6 )
    phi_fn = PhiFnGeneral()
    # phi_fn = PhiFnGeneralExtendedExp()
#    phi_fn = PhiFnStrainHardening(Epp = 0.2, Efp = 0.6, Dfp = 0.2, Elimit = 4.0)
#    phi_fn = PhiFnStrainHardeningBezier()
    # phi_fn = PhiFnStrainHardeningLinear()
    phi_fn.configure_traits()
