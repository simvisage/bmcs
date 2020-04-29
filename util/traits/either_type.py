#-------------------------------------------------------------------------
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
# Created on Aug 7, 2009 by: rchx

from traits.api import TraitType, HasTraits, TraitError
from traitsui.api import View, Item, InstanceEditor
from traitsui.instance_choice import \
    InstanceFactoryChoice


class EitherType(TraitType):

    def __init__(self, names=None, klasses=None, **metadata):
        # validate that these are trait types
        self._klasses = klasses
        self._names = names
        super(EitherType, self).__init__(**metadata)

    def validate(self, object, name, value):
        ''' Set the trait value '''
        # first check if the value is a class
        if isinstance(value, type):
            klass = value
            if not klass in self._klasses:
                raise TraitError('type %s not in the type scope' % klass)
            # check if the last instance of the klass has been
            # registered earlier in the trait history
            new_value = klass()
        else:
            # the value must be one of those in _klasses
            if isinstance(value, tuple(self._klasses)):
                new_value = value
            else:
                raise TraitError('value of type %s out of the scope: %s' %
                                 (value.__class__, self._klasses))
        return new_value

    def get_default_value(self):
        '''Take the first class to construct the value'''
        klass = self._klasses[0]
        value = klass()
        return (0, value)

    def create_editor(self):

        if self._names:
            choice_list = [InstanceFactoryChoice(name=n, klass=k)
                           for n, k in zip(self._names, self._klasses)]
        else:
            choice_list = [InstanceFactoryChoice(klass=k)
                           for k in self._klasses]

        return InstanceEditor(values=choice_list, kind='live')
