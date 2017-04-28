'''
Created on 09.03.2017

@author: cthoennessen
'''
from pyface.api import GUI, PythonEditor
from pyface.tasks.api import \
    TaskWindow, Task, TaskPane, TraitsDockPane, TaskLayout, PaneItem
from traits.api import Instance, Str
from traitsui.api import View


class ExamplePane(TraitsDockPane):

    id = Str
    name = Str

    view = View(
        width=70,
        height=70,
    )


class PythonEditorPane(TaskPane):

    id = Str
    name = Str

    editor = Instance(PythonEditor)

    def create(self, parent):
        self.editor = PythonEditor(parent)
        self.control = self.editor.control

    def destroy(self):
        self.editor.destroy()
        self.control = self.editor = None


class ExampleTask(Task):

    def _default_layout_default(self):
        return TaskLayout(
            top=PaneItem('example_pane_1'),
            right=PaneItem('example_pane_2'),
            bottom=PaneItem('example_pane_3'),
            left=PaneItem('example_pane_4'),
        )

    def activated(self):
        self.window.title = 'Dock Example Task'

    def create_central_pane(self):
        return PythonEditorPane(id='example_editor_pane', name='Python Editor')

    def create_dock_panes(self):
        pane1 = ExamplePane(id='example_pane_1', name='Example Pane 1')
        pane2 = ExamplePane(id='example_pane_2', name='Example Pane 2')
        pane3 = ExamplePane(id='example_pane_3', name='Example Pane 3')
        pane4 = ExamplePane(id='example_pane_4', name='Example Pane 4')
        return [pane1, pane2, pane3, pane4]

if __name__ == '__main__':

    gui = GUI()

    task = ExampleTask()
    window = TaskWindow()
    window.add_task(task)
    window.open()

    gui.start_event_loop()
