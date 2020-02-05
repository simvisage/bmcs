#!/usr/bin/env python
""" The entry point for an Envisage application. """

# Standard library imports.
import sys
import os.path
import logging

# Enthought library imports.
#from etsproxy.mayavi.plugins.app import get_plugins, setup_logger
from etsproxy.mayavi.plugins.app import setup_logger
from traits.api import List, Instance
from etsproxy.envisage.api import Plugin, ServiceOffer, ExtensionPoint
from etsproxy.pyface.workbench.api import Perspective, PerspectiveItem

logger = logging.getLogger()

# View IDs.
PROMOD_VIEW = 'ibvpy.plugins.promod_service.promod_service'
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
        PerspectiveItem(id = PROMOD_VIEW, position = 'top'),
        PerspectiveItem(id = TSTEPPER_VIEW, position = 'bottom'),
        ]

###############################################################################
# `ProModPlugin` class.
###############################################################################
class ProModUIPlugin(Plugin):

    # Extension points we contribute to.
    PERSPECTIVES = 'etsproxy.envisage.ui.workbench.perspectives'
    VIEWS = 'etsproxy.envisage.ui.workbench.views'

    # The plugin's unique identifier.
    id = 'promod_service.promod_service'

    # The plugin's name (suitable for displaying to the user).
    name = 'Product Model'

    # Perspectives.
    perspectives = List(contributes_to = PERSPECTIVES)

    # Views.
    views = List(contributes_to = VIEWS)

    ######################################################################
    # Private methods.
    def _perspectives_default(self):
        """ Trait initializer. """
        return [ProModSpecifyPerspective, ProModAnalyzePerspective]

    def _views_default(self):
        """ Trait initializer. """
        return [self._promod_service_view_factory]

    def _promod_service_view_factory(self, window, **traits):
        """ Factory method for promod_service views. """
        from etsproxy.pyface.workbench.traits_ui_view import \
                TraitsUIView

        promod_service = self._get_promod_service(window)
        tui_engine_view = TraitsUIView(obj = promod_service,
                                       id = 'promod.plugins.promod_service.promod_service',
                                       name = 'Product Model',
                                       window = window,
                                       position = 'left',
                                       **traits
                                       )
        return tui_engine_view

    def _get_promod_service(self, window):
        """Return the promod_service service."""
        return window.get_service('matresdev.db.plugins.promod_service.ProModService')
