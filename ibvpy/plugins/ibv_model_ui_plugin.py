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

# View IDs.
IBVMODEL_VIEW = 'ibvpy.plugins.ibv_model_service.ibv_model_service'
TSTEPPER_VIEW = 'ibvpy.plugins.tstepper_service.tstepper_service'
TLOOP_VIEW = 'ibvpy.plugins.tloop_service.tloop_service'
RTRACEMNGR_VIEW = 'ibvpy.plugins.rtrace_service.rtrace_service'

###############################################################################
# `IBVPPerspective` class.
###############################################################################


class IBVModelSpecifyPerspective(Perspective):

    """ An default perspective for the app. """

    # The perspective's name.
    name = 'Specify IBV Model'

    # Should this perspective be enabled or not?
    enabled = True

    # Should the editor area be shown in this perspective?
    show_editor_area = True

    # The contents of the perspective.
    contents = [
        PerspectiveItem(id=IBVMODEL_VIEW, position='top'),
        PerspectiveItem(id=TSTEPPER_VIEW, position='bottom'),
    ]

###############################################################################
# `IBVPPerspective` class.
###############################################################################


class IBVModelAnalyzePerspective(Perspective):

    """ An default perspective for the app. """

    # The perspective's name.
    name = 'Analyze IBV Model'

    # Should this perspective be enabled or not?
    enabled = True

    # Should the editor area be shown in this perspective?
    show_editor_area = True

    # The contents of the perspective.
    contents = [
        PerspectiveItem(id=IBVMODEL_VIEW, position='top'),
        PerspectiveItem(id=RTRACEMNGR_VIEW,
                        relative_to=IBVMODEL_VIEW,
                        position='bottom'),
        PerspectiveItem(id=TLOOP_VIEW,
                        relative_to=RTRACEMNGR_VIEW,
                        position='bottom'),
    ]


###############################################################################
# `IBVPPlugin` class.
###############################################################################
class IBVModelUIPlugin(Plugin):

    # Extension points we contribute to.
    PERSPECTIVES = 'enthought.envisage.ui.workbench.perspectives'
    VIEWS = 'enthought.envisage.ui.workbench.views'

    # The plugin's unique identifier.
    id = 'ibv_model_service.ibv_model_service'

    # The plugin's name (suitable for displaying to the user).
    name = 'IBV Model'

    # Perspectives.
    perspectives = List(contributes_to=PERSPECTIVES)

    # Views.
    views = List(contributes_to=VIEWS)

    ######################################################################
    # Private methods.
    def _perspectives_default(self):
        """ Trait initializer. """
        return [IBVModelSpecifyPerspective, IBVModelAnalyzePerspective]

    def _views_default(self):
        """ Trait initializer. """
        return [self._ibv_model_service_view_factory]

    def _ibv_model_service_view_factory(self, window, **traits):
        """ Factory method for ibv_model_service views. """
        from pyface.workbench.traits_ui_view import \
            TraitsUIView

        ibv_model_service = self._get_ibv_model_service(window)
        tui_engine_view = TraitsUIView(obj=ibv_model_service,
                                       id='ibvpy.plugins.ibv_model_service.ibv_model_service',
                                       name='IBV Model',
                                       window=window,
                                       position='left',
                                       **traits
                                       )
        return tui_engine_view

    def _get_ibv_model_service(self, window):
        """Return the ibv_model_service service."""
        return window.get_service('ibvpy.plugins.ibv_model_service.IBVModelService')
