'''
Created on 14. 4. 2014

@author: Vancikv
'''

import pickle

from bmcs.utils import \
    get_outfile
from pyface.api import ImageResource
from traits.api import \
    Button, Instance, WeakRef
from traitsui.api import \
    View, Item, HGroup, Handler, \
    UIInfo, spring
from traitsui.file_dialog import \
    open_file, save_file
from traitsui.menu import \
    Action


plot_self = Action(name='Plot', action='plot_node')
'''Context menu action for plotting tree nodes
'''
new_material = Action(name='New material', action='new_material')
'''Context menu action for adding a new database member
'''
del_material = Action(name='Delete', action='del_material')
'''Context menu action for deleting a database member
'''
menu_save = Action(name='Save', action='menu_save')
'''Menubar action for saving the root node to file
'''
menu_open = Action(name='Open', action='menu_open')
'''Menubar action for loading root node from file
'''
menu_exit = Action(name='Exit', action='menu_exit')
'''Menubar action for terminating the view
'''


class BMCSTreeViewHandler(Handler):

    '''Handler for BMCSTreeView class
    '''
    # The UIInfo object associated with the view:
    info = Instance(UIInfo)
    node = WeakRef

    ok = Button('OK')
    cancel = Button('Cancel')
    delete = Button('OK')
    exit_dialog = ('Do you really wish to end '
                   'the session? Any unsaved data '
                   'will be lost.')

    exit_view = View(Item(name='', label=exit_dialog),
                     HGroup(Item('ok', show_label=False, springy=True),
                            Item('cancel', show_label=False, springy=True)
                            ),
                     title='Exit dialog',
                     kind='live'
                     )

    del_view = View(
        HGroup(
            spring,
            Item(name='', label='Really delete?'),
            Item('delete', show_label=False),
            Item('cancel', show_label=False),
        ),
        kind='popup'
    )

    def new_material(self, info, node):
        mat_name = 'new_material_1'
        count = 1
        while node.get(mat_name, None):
            count += 1
            mat_name = 'new_material_' + str(count)
        node[mat_name] = node.klass()

    def del_material(self, info, node):
        if info.initialized:
            self.node = node
            self._ui = self.edit_traits(view='del_view')

    def plot_node(self, info, node):
        '''Handles context menu action Plot for tree nodes
        '''
        info.object.figure.clear()
        node.plot(info.object.figure)
        info.object.data_changed = True

    def menu_save(self, info):
        file_name = get_outfile(folder_name='.bmcs', file_name='')
        file_ = save_file(file_name=file_name)
        if file_:
            pickle.dump(info.object.root, open(file_, 'wb'), 1)

    def menu_open(self, info):
        file_name = get_outfile(folder_name='.bmcs', file_name='')
        file_ = open_file(file_name=file_name)
        if file_:
            info.object.root = pickle.load(open(file_, 'rb'))

    def menu_exit(self, info):
        if info.initialized:
            self.info = info
            self._ui = self.edit_traits(view='exit_view')

    def _delete_fired(self):
        del self.node.db[self.node.key]
        self._ui.dispose()

    def _ok_fired(self):
        self._ui.dispose()
        self.info.ui.dispose()

    def _cancel_fired(self):
        self._ui.dispose()

    #=========================================================================
    # Toolbar actions
    #=========================================================================

    def run(self, info):
        print 'Running action'
        info.object.run()

    def interrupt(self, info):
        print 'Running interrupt'
        info.object.interrupt()

    def continue_(self, info):
        print 'Running continue'
        info.object.continue_()

    def replot(self, info):
        print 'Running continue'
        info.object.replot()

    def clear(self, info):
        info.object.clear()

    def anim(self, info):
        info.object.anim()

    def render(self, info):
        info.object.render()

    def save(self, info):
        info.object.save()

    def load(self, info):
        info.object.load()

action_strings = \
    [('Plot', 'replot', 'Replot current diagrams'),
     ('Clear', 'clear', 'Clear current diagrams'),
     ('Save', 'save', 'Save session'),
     ('Load', 'load', 'Load session'),
     ('Animate', 'anim', 'Animate current session'),
     ('Render', 'render', 'Render current session')]

toolbar_actions = [Action(name="Run",
                          tooltip='Start computation',
                          image=ImageResource('kt-start'),
                          action="run"),
                   Action(name="Pause",
                          tooltip='Pause computation',
                          image=ImageResource('kt-pause'),
                          action="pause"),
                   Action(name="Stop",
                          tooltip='Stop computation',
                          image=ImageResource('kt-stop'),
                          action="stop"),
                   ]

toolbar_actions += [Action(name=name,
                           action=action,
                           tooltip=tooltip)
                    for name, action, tooltip in action_strings]
