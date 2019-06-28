#!/usr/bin/env python
""" The entry point for an Envisage application. """

# Standard library imports.
import logging

# Enthought library imports.
#from etsproxy.mayavi.plugins.app import get_plugins, setup_logger
from traits.api import List
from envisage.api import Plugin, ServiceOffer

logger = logging.getLogger()

###############################################################################
# `RTracePlugin` class.
###############################################################################


class RTracePlugin(Plugin):

    # Extension points we contribute to.
    SERVICE_OFFERS = 'enthought.envisage.ui.workbench.service_offers'

    # The plugin's unique identifier.
    id = 'rtrace_service.rtrace_service'

    # The plugin's name (suitable for displaying to the user).
    name = 'RTrace Manager'

    # Services we contribute.
    service_offers = List(contributes_to=SERVICE_OFFERS)

    ######################################################################
    # Private methods.
    def _service_offers_default(self):
        """ Trait initializer. """
        rtrace_service_service_offer = ServiceOffer(
            protocol='ibvpy.plugins.rtrace_service.RTraceService',
            factory=self._create_rtrace_service
        )
        return [rtrace_service_service_offer]

    def _create_rtrace_service(self, **properties):
        app = self.application
        rtrace_service = app.get_service(
            'ibvpy.plugins.rtrace_service.RTraceService')
        if rtrace_service == None:
            from .rtrace_service import RTraceService
            rtrace_service = RTraceService()
        # attach the service to the current window
        rtrace_service.window = properties['window']
        return rtrace_service

    def _get_rtrace_service(self, window):
        """Return the rtrace_service service."""
        return window.get_service('rtrace_service.RTraceService')
