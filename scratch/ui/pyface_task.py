'''
Created on Apr 20, 2019

@author: rch
'''

from pyface.api import PythonEditor
from pyface.tasks.api import Task
from pyface.tasks.api import TaskPane
from pyface.tasks.api import TraitsDockPane
from traits.api import Event, File, List, Str
from traits.api import Instance
from traitsui.api import View, Item, FileEditor


class ExampleTask(Task):

    id = 'example.example_task'
    name = 'Python Script Editor'

    def create_central_pane(self):
        return PythonEditorPane()

    def create_dock_panes(self):
        """ Create the file browser and connect to its double click event.
        """
        browser = PythonScriptBrowserPane()

        def handler(): return self.open_file(browser.selected_file)
        browser.on_trait_change(handler, 'activated')
        return [browser]

    def open_file(self, filename):
        """ Open the file with the specified path in the central pane.
        """
        self.window.central_pane.editor.path = filename


class PythonEditorPane(TaskPane):

    id = 'example.python_editor_pane'
    name = 'Python Editor'

    editor = Instance(PythonEditor)

    def create(self, parent):
        self.editor = PythonEditor(parent)
        self.control = self.editor.control

    def destroy(self):
        self.editor.destroy()
        self.control = self.editor = None


class FileBrowserPane(TraitsDockPane):

    #### TaskPane interface ###############################################

    id = 'example.file_browser_pane'
    name = 'File Browser'

    #### FileBrowserPane interface ########################################

    # Fired when a file is double-clicked.
    activated = Event

    # The list of wildcard filters for filenames.
    filters = List(Str)

    # The currently selected file.
    selected_file = File

    # The view used to construct the dock pane's widget.
    view = View(Item('selected_file',
                     editor=FileEditor(dclick_name='activated',
                                       filter_name='filters'),
                     style='custom',
                     show_label=False),
                resizable=True)
