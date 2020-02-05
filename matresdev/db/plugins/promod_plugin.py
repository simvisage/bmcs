#!/usr/bin/env python
""" The entry point for an Envisage application. """

# Standard library imports.
import sys
import os.path
import logging

# Enthought library imports.
from etsproxy.mayavi.plugins.app import get_plugins, setup_logger
from traits.api import List
from etsproxy.envisage.api import Plugin, ServiceOffer
from etsproxy.envisage.ui.workbench.api import WorkbenchApplication
from etsproxy.pyface.workbench.api import Perspective, PerspectiveItem

###############################################################################
# `ProModPlugin` class.
###############################################################################
class ProModPlugin(Plugin):

    # Extension points we contribute to.
    SERVICE_OFFERS = 'enthought.envisage.ui.workbench.service_offers'

    # The plugin's unique identifier.
    id = 'ProMod.ProMod'

    # The plugin's name (suitable for displaying to the user).
    name = 'ProMod'

    # Services we contribute.
    service_offers = List(contributes_to = SERVICE_OFFERS)
    
    ######################################################################
    # Private methods.
    def _service_offers_default(self):
        """ Trait initializer. """
        promod_service_offer = ServiceOffer(
            protocol = 'promod.plugins.promod_service.ProModService',
            factory = 'promod.plugins.promod_service.ProModService'
        )

        return [promod_service_offer]
