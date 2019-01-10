
from traits.api import \
    Instance,  \
    List
from traitsui.api import \
    TableEditor, ObjectColumn

from ibvpy.core.i_bcond import IBCond
from view.ui import BMCSTreeNode


# The definition of the demo TableEditor:
bcond_list_editor = TableEditor(
    columns=[ObjectColumn(label='Type', name='var'),
             ObjectColumn(label='Value', name='value'),
             ObjectColumn(label='DOF', name='dof')
             ],
    editable=False,
    selected='object.selected_bcond',
)


class BCondMngr(BMCSTreeNode):

    node_name = 'boundary conditions'

    bcond_list = List(IBCond)

    def _tree_node_list_default(self):
        return self.bcond_list

    selected_bcond = Instance(IBCond)

    def setup(self, sctx):
        '''
        '''
        for bc in self.bcond_list:
            bc.setup(sctx)

    def apply_essential(self, K):
        '''Register the boundary condition in the equation system.
        '''
        for bcond in self.bcond_list:
            bcond.apply_essential(K)

    def apply(self, step_flag, sctx, K, R, t_n, t_n1):

        for bcond in self.bcond_list:
            bcond.apply(step_flag, sctx, K, R, t_n, t_n1)

#     traits_view = View(VSplit(Item('bcond_list', style='custom', editor=bcond_list_editor,
#                                    show_label=False),
#                               Item('selected_bcond@', show_label=False)),
#                        resizable=True,
#                        kind='subpanel',
#                        )


if __name__ == '__main__':
    from ibvpy.api import BCDof
    bc = BCondMngr(bcond_list=[BCDof(), BCDof()])
    print(bc)
    bc.configure_traits()
