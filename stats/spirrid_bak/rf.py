
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Jun 2, 2010 by: rch

from etsproxy.traits.api import HasTraits, List, Str, Property, Event, \
    on_trait_change, cached_property, Tuple
from etsproxy.traits.ui.api import View, Item, VGroup
from inspect import getargspec
import numpy as np
import string


class RF(HasTraits):

    qname = Property
    def _get_qname(self):
        return self.__class__.__name__

    comment = Property
    @cached_property
    def _get_comment(self):
        return self.__doc__

    rf_args = Property(Tuple)
    @cached_property
    def _get_rf_args(self):
        '''
        Extract the traits that are floating points and can be associated 
        with a statistical distribution.
        '''
        # Get the parameters of the obligatory call method

        argspec_list = getargspec(self.__call__).args[1:]

        # this line extracts the traits having the 'distr' metadata

        # containers for control variables
        ctrl_trait_keys = self.traits(ctrl_range = lambda x: x != None)
        ctrl_keys = []
        ctrl_traits = []
        ctrl_values = []

        # containers for parameters
        param_trait_keys = self.traits(distr = lambda x: x != None)
        param_keys = []
        param_traits = []
        param_values = []

        # iterate through the arguments of the call function to get
        # and store them in the corresponding list.
        #
        for argspec in argspec_list:
            if argspec in param_trait_keys:
                param_keys.append(argspec)
                param_traits.append(self.trait(argspec))
                param_values.append(getattr(self, argspec))
            elif argspec in ctrl_trait_keys:
                ctrl_keys.append(argspec)
                ctrl_traits.append(self.trait(argspec))
                ctrl_values.append(getattr(self, argspec))
            else:
                raise RuntimeError('parameter %s not declared as a trait in the response function %s' % \
                    (argspec, self.__class__))
        return (ctrl_keys, ctrl_traits, ctrl_values,
                param_keys, param_traits, param_values)

    #--------------------------------------------------------------------
    # FUNCTION PARAMETERS
    #--------------------------------------------------------------------
    # The declaration of parameters that can be randomized
    # (design parameters)
    param_keys = Property(List)
    def _get_param_keys(self):
        return self.rf_args[3]

    param_traits = Property(List)
    def _get_param_traits(self):
        return self.rf_args[4]

    param_values = Property(List)
    def _get_param_values(self):
        return self.rf_args[5]

    #--------------------------------------------------------------------
    # FUNCTION CONTROL VARIABLES
    #--------------------------------------------------------------------
    ctrl_keys = Property(List)
    def _get_ctrl_keys(self):
        return self.rf_args[0]

    ctrl_traits = Property(List)
    def _get_ctrl_traits(self):
        return self.rf_args[1]

    ctrl_values = Property(List)
    def _get_ctrl_values(self):
        return self.rf_args[2]

    changed = Event
    @on_trait_change('+distr')
    def _set_changed(self):
        self.changed = True

    #@todo: delete - this is motivated by views and interactive editing
    # - shall be done later.

    listener_string = Str('')

    def add_listeners(self):
        self.on_trait_change(self.get_value, self.listener_string)

    def remove_listeners(self):
        self.on_trait_change(self.get_value, self.listener_string, remove = True)

    def default_traits_view(self):
        '''
        Generates the view from the param items.
        '''
        param_items = [ Item(name) for name in self.param_keys ]
        ctrl_items = [ Item(name) for name in self.ctrl_keys ]
        view = View(VGroup(*param_items,
                             id = 'stats.spirrid_bak.rf.params'
                             ),
                    VGroup(*ctrl_items,
                            id = 'stats.spirrid_bak.rf.ctrl'
                             ),
                    kind = 'modal',
                    height = 0.3, width = 0.2,
                    scrollable = True,
                    resizable = True,
                    buttons = ['OK', 'Cancel'],
                    id = 'stats.spirrid_bak.rf'
                    )
        return view

    def plot(self, p, ctrl_idx = 0, **kw):
        X = np.linspace(*self.ctrl_traits[ctrl_idx].ctrl_range)
        Y = self(X, *self.param_values)
        p.plot(X, Y, **kw)
        p.xlabel(self.x_label)
        p.ylabel(self.y_label)
        p.legend(loc = 'best')
        p.title(self.title)

    def plot3d(self, p, ctrl_idx = [0, 1], **kw):
        X = np.linspace(*self.ctrl_traits[ctrl_idx[0]].ctrl_range)
        Y = np.linspace(*self.ctrl_traits[ctrl_idx[1]].ctrl_range)
        Z = self(X, Y, *self.param_values)
        p.surf(X, Y, Z, **kw)

    def __str__(self):
        ctrl_list = [ '%s' % nm
                     for nm in self.ctrl_keys]
        param_list = [ '%s = %g' % (nm, v)
                     for nm, v in zip(self.param_keys, self.param_values)]
        ctrl = string.join(ctrl_list, ', ')
        params = string.join(param_list, ', ')
        return '%s\n%s' % (ctrl, params)
