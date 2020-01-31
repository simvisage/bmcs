'''
Created on Nov 12, 2009

@author: rch
'''
from math import pi

from traits.api import HasTraits, Float, Property, cached_property
from traitsui.api import View, Item, HGroup, VGroup, Group


class SimplyRatio(HasTraits):
    '''
    Class specifying explicitly the reinforcement. 
    '''
    rho = Float(0.032, auto_set=False, enter_set=True,  # [-]
                desc='the reinforcement ratio [-]',
                modified=True)

    traits_view = View(Item('rho',
                            label='reinforcement ratio'),
                       resizable=True,
                       )


class GridReinforcement(HasTraits):
    '''
    Class delivering reinforcement ratio for a grid reinforcement of a cross section
    '''

    h = Float(30, auto_set=False, enter_set=True,  # [mm]
              desc='the height of the cross section',
              modified=True)
    w = Float(100, auto_set=False, enter_set=True,  # [mm]
              desc='the width of the cross section',
              modified=True)
    n_f_h = Float(9, auto_set=False, enter_set=True,  # [-]
                  desc='the number of fibers in the height direction',
                  modified=True)
    n_f_w = Float(12, auto_set=False, enter_set=True,  # [-]
                  desc='the number of fibers in the width direction',
                  modified=True)
    a_f = Float(0.89, auto_set=False, enter_set=True,  # [m]
                desc='the cross sectional area of a single fiber',
                modified=True)

    A_tot = Property(Float,
                     depends_on='+modified')

    def _get_A_tot(self):
        return self.h * self.w

    A_f = Property(Float,
                   depends_on='+modified')

    def _get_A_f(self):
        n_f = self.n_f_h * self.n_f_w
        a_f = self.a_f
        return a_f * n_f

    rho = Property(Float,
                   depends_on='+modified')

    @cached_property
    def _get_rho(self):
        return self.A_f / self.A_tot

    traits_view = View(VGroup(
        Group(
            Item('h', label='height', full_size=True, resizable=True),
            Item('w', label='width', resizable=True),
            label='cross section dimensions',
            orientation='vertical'
        ),
        Group(
            Item('a_f',   label='area of a single fiber',
                 full_size=True, resizable=True),
            Item('n_f_h', label='# in height direction', resizable=True),
            Item('n_f_w', label='# in width direction', resizable=True),
            label='layout of the fiber grid',
            orientation='vertical'
        ),
        Item('rho', label='current reinforcement ratio',
             style='readonly', emphasized=True),
        #                           label = 'Cross section parameters',
        id='scm.cs.params',
    ),
        id='scm.cs',
        dock='horizontal',
        resizable=True,
        height=0.8, width=0.8
    )


if __name__ == '__main__':
    cs = GridReinforcement()
    cs.configure_traits()
