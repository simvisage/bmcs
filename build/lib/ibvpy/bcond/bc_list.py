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
# Created on May 26, 2009 by: rchx

from traits.api import  \
    Str, Enum, \
    List, cached_property, \
    provides, Property
from traitsui.api import \
    VSplit, \
    View, UItem, Item, TableEditor, VGroup

from ibvpy.core.i_bcond import \
    IBCond
from traitsui.table_column \
    import ObjectColumn
from view.plot2d import Vis2D
from view.ui import BMCSTreeNode

from .bc_dof import BCDof


bcond_list_editor = TableEditor(
    columns=[ObjectColumn(label='Type', name='var'),
             ObjectColumn(label='Value', name='value'),
             ObjectColumn(label='DOF', name='dof')
             ],
    editable=False,
)


@provides(IBCond)
class BCDofList(BMCSTreeNode, Vis2D):
    '''
    Implements the IBC functionality for a constrained dof.
    '''
    tree_node_list = List

    tree_node_list = Property(depends_on='bcdof_list,bcdof_list_items')

    @cached_property
    def _get_tree_node_list(self):
        return self.bcdof_list

    name = Str('<unnamed>')

    node_name = Property

    def _get_node_name(self):
        s = '%s:%s=%s' % (self.var, self.slice, self.value)
        return s

    var = Enum('u', 'f', 'eps', 'sig')

    def is_essential(self):
        return self.var == 'u'

    def is_linked(self):

        return self.link_dofs != []

    def is_constrained(self):
        '''
        Return true if a DOF is either explicitly prescribed or it depends on other DOFS.
        '''
        return self.is_essential() or self.is_linked()

    def is_natural(self):
        return self.var == 'f' or self.var == 'eps' or self.var == 'sig'

    bcdof_list = List(BCDof)

    def reset(self):
        self.bcdof_list = []

    integ_domain = Enum(['global', 'local'])

    def setup(self, sctx):
        '''
        Locate the spatial context.f
        '''

    def apply_essential(self, K):

        for bcond in self.bcdof_list:
            bcond.apply_essential(K)

    def apply(self, step_flag, sctx, K, R, t_n, t_n1):

        if self.is_essential():
            for bcond in self.bcdof_list:
                bcond.apply(step_flag, sctx, K, R, t_n, t_n1)
        else:
            self.apply_natural(step_flag, sctx, K, R, t_n, t_n1)

    def apply_natural(self, step_flag, sctx, K, R, t_n, t_n1):

        raise NotImplementedError

    traits_view = View(
        VGroup(
            VSplit(
                Item('bcdof_list',
                     style='custom',
                     editor=bcond_list_editor,
                     show_label=False),
            ),
        )
    )

    tree_view = traits_view
