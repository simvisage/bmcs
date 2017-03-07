'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    HasStrictTraits, Dict, Property, Float, \
    WeakRef, DelegatesTo, cached_property, \
    Str, List, Button
from traitsui.api import \
    View, Group, UItem, Include, EnumEditor, HGroup

from view.ui import BMCSLeafNode


class Viz2DDict(HasStrictTraits):
    '''On demand constructor of viz2d object, 
    Objects are constructed upon access using the key within  
    the viz2d_classes dictionary.
    '''

    vis2d = WeakRef

    viz2d_classes = DelegatesTo('vis2d')

    _viz2d_objects = Dict

    def __getitem__(self, key):
        viz2d = self._viz2d_objects.get(key, None)
        if viz2d == None:
            viz2d_class = self.viz2d_classes.get(key, None)
            if viz2d_class == None:
                raise KeyError, 'No vizualization class with key %s' % key
            viz2d = viz2d_class(label=key, vis2d=self.vis2d)
            self._viz2d_objects[key] = viz2d
        return viz2d

    def __delitem__(self, key):
        del self._viz2d_objects[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self._viz2d_objects)

    def __len__(self):
        return len(self._viz2d_objects)

    def __keytransform__(self, key):
        return key

    def __repr__(self):
        return repr(self._viz2d_objects)


class Vis2D(BMCSLeafNode):
    '''Each state and operator object can be associated with 
    several visualization objects with a shortened class name Viz3D. 
    In order to introduce a n independent class subsystem into 
    the class structure, objects supporting visualization inherit 
    from Visual3D which introduces a dictionary viz3d objects.
    '''

    vot = Float(0.0, time_change=True)
    '''Visual object time
    '''

    viz2d_classes = Dict
    '''Visualization classes applicable to this object. 
    '''

    viz2d_class_names = Property(List(Str),
                                 depends_on='viz2d_classes')
    '''Keys of the viz2d classes
    '''
    @cached_property
    def _get_viz2d_class_names(self):
        return self.viz2d_classes.keys()

    selected_viz2d_class = Str('default')

    add_selected_viz2d = Button(label='Add plot viz2d')

    def _add_selected_viz2d_fired(self):
        viz2d = self.viz2d[self.selected_viz2d_class]
        if self.ui:
            self.ui.plot_dock_pane.viz2d_list.append(viz2d)

    viz2d = Property(Dict)
    '''Dictionary of visualization objects'''
    @cached_property
    def _get_viz2d(self):
        '''Get a vizualization object given the key
        of the vizualization class. Construct it on demand
        and register in the viz3d_dict.
        '''
        return Viz2DDict(vis2d=self)

    def viz2d_notify_change(self):
        for viz2d in self.viz2d.values():
            viz2d.vis2d_changed = True

    actions = Group(
        HGroup(
            UItem('add_selected_viz2d'),
            UItem('selected_viz2d_class', springy=True,
                  editor=EnumEditor(name='object.viz2d_class_names',
                                    )
                  ),
            springy=True,
        ),
    )

    view = View(
        Include('actions'),
        resizable=True
    )

if __name__ == '__main__':

    from example import rt
    rt.configure_traits()
