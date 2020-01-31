

import os
import pickle

from bmcs.view.window import BMCSWindow
from ibvpy.api import BCDof
from traits.etsconfig.api import ETSConfig

from pull_out_simulation import\
    LoadingScenario, Geometry, PullOutSimulation


if ETSConfig.toolkit == 'wx':
    from traitsui.wx.tree_editor import \
        NewAction, DeleteAction, CopyAction, PasteAction
if ETSConfig.toolkit == 'qt4':
    from traitsui.qt4.tree_editor import \
        NewAction, DeleteAction, CopyAction, PasteAction
else:
    raise ImportError, "tree actions for %s toolkit not availabe" % \
        ETSConfig.toolkit

# =========================================================================
# List of all custom nodes
# =========================================================================

loading_scenario = LoadingScenario()

bc_list = [BCDof(node_name='fixed left end', var='u',
                 dof=0, value=0.0),
           BCDof(node_name='pull-out displacement', var='u', dof=-1,
                 time_function=loading_scenario.time_func)]

#loading_scenario = LoadingScenario()
geometry = Geometry()

model = PullOutSimulation(
    geometry=geometry, loading_scenario=loading_scenario)

model.time_stepper.bcond_list = bc_list

w = BMCSWindow(root=model)
w.configure_traits()
