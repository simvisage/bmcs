'''
Created on 14. 4. 2014

@author: rch
'''

import string

from reporter import RInputRecord
from traits.api import \
    HasStrictTraits, Str, List, WeakRef, \
    Property, cached_property, on_trait_change, Event
from traitsui.api import \
    View


itags = dict(
    TIME=True,
    ALG=True,
    GEO=True,
    MESH=True,
    MAT=True,
    FE=True,
    CS=True,
    BC=True
)

itags_str = ','.join(['%s' % tag for tag in itags])


class BMCSNodeBase(HasStrictTraits):

    node_name = Str('<unnamed>')

    tree_view = View()

    ui = WeakRef

    def set_traits_with_metadata(self, value, **metadata):
        traits_names = self.trait_names(**metadata)
        for tname in traits_names:
            setattr(self, tname, value)

    parent = WeakRef

    root = Property
    '''Root node of tree node hierarchy
    '''

    def _get_root(self):
        if self.parent:
            return self.parent.root
        return self


class BMCSListeningTreeNodeMixIn(HasStrictTraits):

    @on_trait_change('+TIME')
    def _TIME_change(self):
        if self.parent:
            # print 'TIME change'
            self.root.TIME = True

    @on_trait_change('+ALG')
    def _ALG_change(self):
        if self.parent:
            # print 'ALG change'
            self.root.ALG = True

    @on_trait_change('+GEO')
    def _GEO_change(self):
        if self.parent:
            # print 'GEO change'
            self.root.GEO = True

    @on_trait_change('+MESH')
    def _MESH_change(self):
        if self.parent:
            # print 'MESH change'
            self.root.MESH = True

    @on_trait_change('+MAT')
    def _MAT_change(self):
        if self.parent:
            # print 'MAT change'
            self.root.MAT = True

    @on_trait_change('+FE')
    def _FE_change(self):
        if self.parent:
            # print 'FE change'
            self.root.FE = True

    @on_trait_change('+CS')
    def _CS_change(self):
        if self.parent:
            # print 'CS change'
            self.root.CS = True

    @on_trait_change('+BC')
    def _BC_change(self):
        if self.parent:
            # print 'BC change'
            self.root.BC = True


class BMCSLeafNodeMixIn(HasStrictTraits):
    '''Base class of all model classes that can appear in a tree view.
    '''

    def set_parents_recursively(self):
        return

    def set_ui_recursively(self, ui):
        self.ui = ui


class BMCSTreeNodeMixIn(HasStrictTraits):
    '''Base class of all model classes that can appear in a tree view.
    '''
    tree_node_list = List([])

    def _tree_node_list_items_changed(self):
        self.set_parents_recursively()

    def _tree_node_list_changed(self):
        self.set_parents_recursively()

    def set_parents_recursively(self):
        for n in self.tree_node_list:
            if n is None:
                continue
            n.parent = self
            n.set_parents_recursively()

    def set_ui_recursively(self, ui):
        self.ui = ui
        for node in self.tree_node_list:
            node.set_ui_recursively(ui)

    def append_node(self, node):
        '''Add a new subnode to the current node.
        Inform the tree view to select the new node within the view.
        '''
        node.set_ui_recursively(self.ui)
        self.tree_node_list.append(node)

    def set_traits_with_metadata(self, value, **metadata):
        super(BMCSTreeNodeMixIn, self).set_traits_with_metadata(
            value, **metadata)
        for node in self.tree_node_list:
            node.set_traits_with_metadata(value, **metadata)


class BMCSLeafNode(BMCSNodeBase,
                   RInputRecord,
                   BMCSLeafNodeMixIn,
                   BMCSListeningTreeNodeMixIn):
    itags = itags


class BMCSTreeNode(BMCSNodeBase,
                   RInputRecord,
                   BMCSTreeNodeMixIn,
                   BMCSListeningTreeNodeMixIn):
    itags = itags


class BMCSRootNode(BMCSNodeBase,
                   RInputRecord,
                   BMCSTreeNodeMixIn,
                   BMCSListeningTreeNodeMixIn):
    '''Root node for the model hierarchy.

    Types of generic change events within a numerical simulation

    time discretization
    algorithm parameters
    geometry (dimensions, scaling, geometrical transformation)
    spatial discretization (spatial meshing / decomposition parameters)
    spatial approximation (finite elements - what happens within an element)
    cross section (thickness, area, perimeter)
    boundary conditions

    '''

    itags = itags

    TIME = Event
    ALG = Event
    GEO = Event
    MESH = Event
    MAT = Event
    FE = Event
    CS = Event
    BC = Event

    def __init__(self, *args, **kw):
        super(BMCSRootNode, self).__init__(*args, **kw)
        self.set_parents_recursively()

    @on_trait_change('+TIME')
    def _TIME_change(self):
        # print 'TIME change'
        self.root.TIME = True

    @on_trait_change('+ALG')
    def _ALG_change(self):
        # print 'ALG change'
        self.root.ALG = True

    @on_trait_change('+GEO')
    def _GEO_change(self):
        # print 'GEO change'
        self.root.GEO = True

    @on_trait_change('+MESH')
    def _MESH_change(self):
        # print 'MESH change'
        self.root.MESH = True

    @on_trait_change('+MAT')
    def _MAT_change(self):
        # print 'MAT change'
        self.root.MAT = True

    @on_trait_change('+FE')
    def _FE_change(self):
        # print 'FE change'
        self.root.FE = True

    @on_trait_change('+CS')
    def _CS_change(self):
        # print 'CS change'
        self.root.CS = True

    @on_trait_change('+BC')
    def _BC_change(self):
        # print 'BC change'
        self.root.BC = True
