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
# Created on Jan 19, 2010 by: rch

from numpy import \
    linspace, arange
from traits.api import \
    Bool, Float, HasTraits, \
    Instance, Int, Str, \
    List, \
    Property, cached_property
from traitsui.api import \
    View, Item, TabularEditor, Group
from traitsui.tabular_adapter import \
    TabularAdapter

from .i_sim_model import \
    ISimModel


class LevelListAdapter(TabularAdapter):

    columns = Property

    #---------------------------------------------------------------
    # EXTRACT FACTOR NAMES AS COLUMNS FOR TABLE EDITOR
    #-------------------------------------------------------------------
    def _get_columns(self):
        return [('level', 'index'), ('value', 0)]

    font = 'Courier 10'
    alignment = 'center'
    odd_bg_color = 'lightgray'

    index_width = Float(40)
    index_text = Property

    def _get_index_text(self):
        return str(self.row)

    val_width = Float(100)

level_list_editor = TabularEditor(adapter=LevelListAdapter())


class SimFactor(HasTraits):

    model = Instance(ISimModel)

    unit = Str('-')

    regular = Bool

    levels = List

    level_list = Property(depends_on='+levels_modified')

    @cached_property
    def _get_level_list(self):
        raise NotImplementedError

    levels_table = Property(depends_on='+levels_modified')

    @cached_property
    def _get_levels_table(self):
        return [[level] for level in self.level_list]

    def get_n_levels(self):
        return len(self.level_list)


class SimNumFactor(SimFactor):
    '''
    Parameter definition derived from the Param and SimModel
    '''
    regular = True

    def get_level_value(self, v):
        return v

    traits_view = View(Item('regular'),
                       Group(
        Item('min_level', visible_when='regular'),
        Item('max_level', visible_when='regular'),
        Item('n_levels', visible_when='regular'),
        columns=2,
    ),
        Group(
        Item('levels_table', editor=level_list_editor,
             show_label=False,
             style='custom'),
    ),
        scrollable=True)


class SimIntFactor(SimNumFactor):
    '''
    Parameter definition derived from the Param and SimModel
    '''
    min_level = Int(0, auto_set=False, enter_set=True, levels_modified=True)
    max_level = Int(1, auto_set=False, enter_set=True, levels_modified=True)
    n_levels = Int(1, auto_set=False, enter_set=True, levels_modified=True)

    def _get_level_list(self):
        if self.regular:
            step = (self.max_level - self.min_level) / self.n_levels
            if int(step) == 0:
                step = 1
            levels = arange(
                self.min_level, self.max_level + 1, step, dtype=int)
            return levels
        else:
            return self.levels


class SimFloatFactor(SimNumFactor):
    '''
    Parameter definition derived from the Param and SimModel
    '''
    min_level = Float(0, auto_set=False, enter_set=True, levels_modified=True)
    max_level = Float(1, auto_set=False, enter_set=True, levels_modified=True)
    n_levels = Int(2, auto_set=False, enter_set=True, levels_modified=True)

    def _get_level_list(self):
        if self.regular:
            return linspace(self.min_level, self.max_level, self.n_levels)
        else:
            return self.levels


class SimEnumFactor(SimFactor):
    '''
    Parameter definition derived from the Param and SimModel
    '''
    regular = False
    levels = List(auto_set=False, enter_set=True, levels_modified=True)

    traits_view = View(
        Item('levels_table', editor=level_list_editor,
             show_label=False,
             style='custom'),
        scrollable=True
    )

    def get_level_value(self, v):
        return getattr(self.model, v)

    def _get_level_list(self):
        return self.levels
