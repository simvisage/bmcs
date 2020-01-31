#!/usr/bin/env python
""" The entry point for an Envisage application. """

# Standard library imports.
import logging

# Enthought library imports.
#from etsproxy.mayavi.plugins.app import get_plugins, setup_logger

from traits.api import List
from envisage.api import Plugin
from pyface.workbench.api import Perspective, PerspectiveItem

logger = logging.getLogger()

###############################################################################
# `IBVPPerspective` class.
###############################################################################


class TLoopPerspective(Perspective):

    """ An default perspective for the app. """

    # The perspective's name.
    name = 'Time Loop'

    # Should this perspective be enabled or not?
    enabled = True

    # Should the editor area be shown in this perspective?
    show_editor_area = True

    # View IDs.
    TLOOPMNGR_VIEW = 'ibvpy.plugins.tloop_service.tloop_service'

    # The contents of the perspective.
    contents = [
        PerspectiveItem(id=TLOOPMNGR_VIEW, position='left'),
    ]

###############################################################################
# `IBVPPlugin` class.
###############################################################################


class TLoopUIPlugin(Plugin):

    # Extension points we contribute to.
    PERSPECTIVES = 'enthought.envisage.ui.workbench.perspectives'
    VIEWS = 'enthought.envisage.ui.workbench.views'

    # The plugin's unique identifier.
    id = 'tloop_service.tloop_service'

    # The plugin's name (suitable for displaying to the user).
    name = 'Time loop'

    # Perspectives.
    perspectives = List(contributes_to=PERSPECTIVES)

    # Views.
    views = List(contributes_to=VIEWS)

    ######################################################################
    # Private methods.
    def _perspectives_default(self):
        """ Trait initializer. """
        return [TLoopPerspective]

    def _views_default(self):
        """ Trait initializer. """
        return [self._tloop_service_view_factory]

    def _tloop_service_view_factory(self, window, **traits):
        """ Factory method for tloop_service views. """
        from pyface.workbench.traits_ui_view import \
            TraitsUIView

        tloop_service = self._get_tloop_service(window)
        tui_engine_view = TraitsUIView(obj=tloop_service,
                                       id='ibvpy.plugins.tloop_service.tloop_service',
                                       name='Time loop',
                                       window=window,
                                       position='left',
                                       **traits
                                       )
        return tui_engine_view

    def _get_tloop_service(self, window):
        """Return the tloop_service service."""
        return window.get_service('ibvpy.plugins.tloop_service.TLoopService')
