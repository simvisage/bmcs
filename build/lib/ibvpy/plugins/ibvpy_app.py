"""The Mayavi Envisage application.
"""
# Author: Prabhu Ramachandran <prabhu_r@users.sf.net>
# Copyright (c) 2008, Enthought, Inc.
# License: BSD Style.

# Standard library imports.
import logging
import sys

from mayavi.core.customize import get_custom_plugins
from mayavi.preferences.api import preference_manager
from traits.api import \
    HasTraits, Instance, Int, \
    on_trait_change

from .ibv_model_plugin import IBVModelPlugin
from .ibv_model_ui_plugin import IBVModelUIPlugin
from .ibvpy_workbench_application import IBVPyWorkbenchApplication
import mayavi.plugins.app as mayavi_app
from .mayavi_engine import set_engine
from .rtrace_plugin import RTracePlugin
from .rtrace_ui_plugin import RTraceUIPlugin
from .tloop_plugin import TLoopPlugin
from .tloop_ui_plugin import TLoopUIPlugin
from .tstepper_plugin import TStepperPlugin
from .tstepper_ui_plugin import TStepperUIPlugin


# Enthought library imports.
# Local imports.
# from etsproxy import ETS_BASENAME
ETS_BASENAME = ''

# GLOBALS
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# local imports
#from sdomain_plugin import SDomainPlugin
#from sdomain_ui_plugin import SDomainUIPlugin


def get_plugins():
    """Get list of default plugins to use for IBVPy."""
    plugins = mayavi_app.get_plugins()
    plugins.insert(0, IBVModelPlugin())
    plugins.insert(1, IBVModelUIPlugin())
    plugins.insert(2, TLoopPlugin())
    plugins.insert(3, TLoopUIPlugin())
    plugins.insert(4, TStepperPlugin())
    plugins.insert(5, TStepperUIPlugin())
#    plugins.insert(6, SDomainPlugin() )
#    plugins.insert(7, SDomainUIPlugin() )
    plugins.insert(6, RTracePlugin())
    plugins.insert(7, RTraceUIPlugin())
    return plugins


def get_non_gui_plugins():
    """Get list of basic tloop plugins that do not add any views or
    actions."""
    plugins = mayavi_app.get_plugins()
    plugins.insert(0, TLoopPlugin())
    plugins.insert(1, TStepperPlugin())
    plugins.insert(2, RTracePlugin())
    return plugins

###########################################################################
# `IBVPy` class.
###########################################################################


class IBVPyApp(HasTraits):

    """The IBVPy application class.

    This class may be easily subclassed to do something different.
    For example, one way to script IBVPy (as a standalone application
    and not interactively) is to subclass this and do the needful.
    """

    # The main envisage application.
    application = Instance(ETS_BASENAME +
                           'envisage.ui.workbench.api.WorkbenchApplication')

    # The IBVPy Script instance.
    script = Instance(ETS_BASENAME +
                      'mayavi.plugins.script.Script')

    # TLoop instance constructed by scripting
    # @todo: make a base class for the IBVModel components that should be here
    # instead of Any
    ibv_resource = Instance('ibvpy.core.ibv_resource.IBVResource')

    # The logging mode.
    log_mode = Int(logging.DEBUG, desc='the logging mode to use')

    def main(self, argv=None, plugins=None):
        """The main application is created and launched here.

        Parameters
        ----------

        - argv : `list` of `strings`

          The list of command line arguments.  The default is `None`
          where no command line arguments are parsed.  To support
          command line arguments you can pass `sys.argv[1:]`.

        - plugins : `list` of `Plugin`s

          List of plugins to start.  If none is provided it defaults to
          something meaningful.

        - log_mode : The logging mode to use.

        """
        # Parse any cmd line args.
        if argv is None:
            argv = []
        self.parse_command_line(argv)

        if plugins is None:
            plugins = get_plugins()

        plugins += get_custom_plugins()

        # Create the application
        prefs = preference_manager.preferences
        app = IBVPyWorkbenchApplication(plugins=plugins,
                                        preferences=prefs)
        self.application = app

        # Setup the logger.
        # self.setup_logger()

        # Start the application.
        app.run()

    def setup_logger(self):
        """Setup logging for the application."""
        mayavi_app.setup_logger(logger, 'ibvpy.log', mode=self.log_mode)

    def parse_command_line(self, argv):
        """Parse command line options.

        Parameters
        ----------

        - argv : `list` of `strings`

          The list of command line arguments.
        """
        from optparse import OptionParser
        usage = "usage: %prog [options]"
        parser = OptionParser(usage)

        (options, args) = parser.parse_args(argv)

    def run(self):
        """This function is called after the GUI has started.
        Override this to do whatever you want to do as a IBVPy
        script.  If this is not overridden then an empty IBVPy
        application will be started.

        *Make sure all other IBVPy specific imports are made here!*
        If you import IBVPy related code earlier you will run into
        difficulties.  Use 'self.script' to script the ibvpy engine.
        """
        app = self.application
        if self.ibv_resource:
            window = app.workbench.active_window
            e = window.get_service(ETS_BASENAME +
                                   'mayavi.core.engine.Engine')
            set_engine(e)
            self.ibv_resource.bind_services(window)
            self.ibv_resource.register_mv_pipelines(e)

    ######################################################################
    # Non-public interface.
    ######################################################################
    @on_trait_change('application.gui:started')
    def _on_application_gui_started(self, obj, trait_name, old, new):
        """This is called as soon as  the Envisage GUI starts up.  The
        method is responsible for setting our script instance.
        """
        if trait_name != 'started' or not new:
            return
        app = self.application
        from mayavi.plugins.script import Script
        window = app.workbench.active_window
        # Set our script instance.
        self.script = window.get_service(Script)
        # Call self.run from the GUI thread.
        app.gui.invoke_later(self.run)


def main(argv=None):
    """Simple helper to start up the ibvpy application.  This returns
    the running application."""
    m = IBVPyApp()
    m.main(argv)
    return m


if __name__ == '__main__':
    main(sys.argv[1:])
