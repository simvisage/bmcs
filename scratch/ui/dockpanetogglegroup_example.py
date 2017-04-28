'''
Created on 10.03.2017

@author: cthoennessen
'''
# Enthought library imports.
from pyface.api import GUI
from pyface.tasks.action.api import SMenu, SMenuBar, SGroup, \
    DockPaneToggleGroup
from pyface.tasks.api import DockPane, Task, TaskPane, TaskWindow
from traits.api import List


class BogusTask(Task):

    id = 'tests.bogus_task'
    name = 'Bogus Task'

    def create_central_pane(self):
        return TaskPane(id='tests.bogus_task.central_pane')

    def create_dock_panes(self):
        dock_panes = [
            DockPane(id='tests.bogus_task.dock_pane_1', name='Dock Pane 1'),
            DockPane(id='tests.bogus_task.dock_pane_2', name='Dock Pane 2'),
        ]
        return dock_panes

    def _menu_bar_default(self):
        menu_bar = SMenuBar(
            SMenu(
                SGroup(
                    group_factory=DockPaneToggleGroup,
                    id='tests.bogus_task.DockPaneToggleGroup'
                ),
                id= 'View', name='View'
            )
        )

        return menu_bar

if __name__ == '__main__':
    gui = GUI()

    task = BogusTask()
    window = TaskWindow()
    window.add_task(task)
    window.open()

    gui.start_event_loop()