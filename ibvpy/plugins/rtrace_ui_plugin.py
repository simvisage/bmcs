#!/usr/bin/env python
""" The entry point for an Envisage application. """

# Standard library imports.
import logging

# Enthought library imports.
#from etsproxy.mayavi.plugins.app import get_plugins, setup_logger
from mayavi.plugins.app import setup_logger
from traits.api import List
from envisage.api import Plugin
from pyface.workbench.api import Perspective, PerspectiveItem
from ibvpy.api import RTDofGraph

logger = logging.getLogger()

###############################################################################
# `RTracePerspective` class.
###############################################################################


class RTracePerspective(Perspective):

    """ An default perspective for the app. """

    # The perspective's name.
    name = 'RTrace'

    # Should this perspective be enabled or not?
    enabled = True

    # Should the editor area be shown in this perspective?
    show_editor_area = True

    # View IDs.
    RTRACEMNGR_VIEW = 'ibvpy.plugins.rtrace_service.rtrace_service'

    # The contents of the perspective.
    contents = [
        PerspectiveItem(id=RTRACEMNGR_VIEW, position='left'),
    ]

###############################################################################
# `RTracePlugin` class.
###############################################################################


class RTraceUIPlugin(Plugin):

    # Extension points we contribute to.
    PERSPECTIVES = 'enthought.envisage.ui.workbench.perspectives'
    VIEWS = 'enthought.envisage.ui.workbench.views'

    # The plugin's unique identifier.
    id = 'rtrace_service.rtrace_service'

    # The plugin's name (suitable for displaying to the user).
    name = 'RTraces'

    # Perspectives.
    perspectives = List(contributes_to=PERSPECTIVES)

    # Views.
    views = List(contributes_to=VIEWS)

    ######################################################################
    # Private methods.
    def _perspectives_default(self):
        """ Trait initializer. """
        return [RTracePerspective]

    def _views_default(self):
        """ Trait initializer. """
        return [self._rtrace_service_view_factory]

    def _rtrace_service_view_factory(self, window, **traits):
        """ Factory method for rtrace_service views. """
        from pyface.workbench.traits_ui_view import \
            TraitsUIView

        rtrace_service = self._get_rtrace_service(window)
        tui_engine_view = TraitsUIView(obj=rtrace_service,
                                       id='ibvpy.plugins.rtrace_service.rtrace_service',
                                       name='Response traces',
                                       window=window,
                                       position='left',
                                       **traits
                                       )
        return tui_engine_view

    def _get_rtrace_service(self, window):
        """Return the rtrace_service service."""
        return window.get_service('ibvpy.plugins.rtrace_service.RTraceService')


def get_plugins():
    """Get list of default plugins to use for Mayavi."""
    from envisage.core_plugin import CorePlugin
    from envisage.ui.workbench.workbench_plugin import WorkbenchPlugin
    from envisage.plugins.python_shell.python_shell_plugin import PythonShellPlugin
    from tvtk.plugins.scene.scene_plugin import ScenePlugin
    from tvtk.plugins.scene.ui.scene_ui_plugin import SceneUIPlugin
    from mayavi.plugins.mayavi_plugin import MayaviPlugin
    from mayavi.plugins.mayavi_ui_plugin import MayaviUIPlugin
    from envisage.developer.developer_plugin import DeveloperPlugin
    from envisage.developer.ui.developer_ui_plugin import DeveloperUIPlugin

    plugins = [CorePlugin(),
               WorkbenchPlugin(),
               MayaviPlugin(),
               MayaviUIPlugin(),
               ScenePlugin(),
               SceneUIPlugin(),
               PythonShellPlugin(),
               DeveloperPlugin(),
               DeveloperUIPlugin(),
               #              TextEditorPlugin()
               ]

    return plugins
######################################################################


def main():

    # Get the default mayavi plugins.
    plugins = get_plugins()
    from .rtrace_plugin import RTracePlugin
    from .rtrace_service import RTraceService

    # Inject our plugin up front so our perspective becomes the default.
    #plugins = [ RTracePlugin() ]
    plugins.insert(0, RTracePlugin())

    from .ibvpy_workbench_application import IBVPyWorkbenchApplication
    # Create an Envisage application.
    id = 'rtrace_service.rtrace_service'
    application = IBVPyWorkbenchApplication(id=id, plugins=plugins)

    rtrace_mgr = RTraceService(rtrace_list=[
        RTDofGraph(name='rte 1'),
        RTDofGraph(name='rte 2'),
        RTDofGraph(name='rte 3'),
        RTDofGraph(name='rte 4'),
        RTDofGraph(name='rte 5'),
        RTDofGraph(name='rte 6'),
        RTDofGraph(name='rte 7'),
        RTDofGraph(name='rte 8'),
        RTDofGraph(name='rte 8'),
        RTDofGraph(name='rte 10'),
        RTDofGraph(name='rte 11'),
    ])
    application.register_service('rtrace_service.RTraceService', rtrace_mgr)

    setup_logger(logger, 'rtrace.log', mode=logging.ERROR)

    # Start the application.
    application.run()

# Application entry point.
if __name__ == '__main__':
    main()
