'''
Created on 14. 4. 2014

@author: Vancikv
'''

from traits.api import Range
from traits.api import \
    HasStrictTraits, Str, List, WeakRef, \
    Property, cached_property
from traitsui.api import \
    View


class BMCSLeafNode(HasStrictTraits):
    '''Base class of all model classes that can appear in a tree view.
    '''
    node_name = Str('<unnamed>')

    ui = WeakRef


class BMCSTreeNode(HasStrictTraits):
    '''Base class of all model classes that can appear in a tree view.
    '''
    node_name = Str('<unnamed>')

    tree_node_list = List([])

    ui = WeakRef

    tree_view = View()

    def append_node(self, node):
        '''Add a new subnode to the current node.
        Inform the tree view to select the new node within the view.
        '''
        self.tree_node_list.append(node)


class ReinfLayoutTreeNode(BMCSTreeNode):
    '''Class accommodating the list of all reinforcement components.
    '''
    node_name = Str('Reinforcement layout')

    cs_state = WeakRef(HasStrictTraits)

    def __getstate__(self):
        '''Overriding __getstate__ because of WeakRef usage
        '''
        state = super(HasStrictTraits, self).__getstate__()

        for key in ['cs_state', 'cs_state_']:
            if state.has_key(key):
                del state[key]

        return state

    def plot(self, fig):
        ax = fig.add_subplot(1, 1, 1)
        self.cs_state.plot_geometry(ax)

    tree_node_list = Property(
        depends_on='cs_state.reinf_components_with_state')

    @cached_property
    def _get_tree_node_list(self):
        self.cs_state.changed = True
        return self.cs_state.reinf_components_with_state
