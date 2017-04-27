#!/usr/bin/env python
""" The entry point for an Envisage application. """

# Enthought library imports.
from traits.api import List
from envisage.api import Plugin, ServiceOffer

###############################################################################
# `TLoopPlugin` class.
###############################################################################


class TLoopPlugin(Plugin):

    # Extension points we contribute to.
    SERVICE_OFFERS = 'enthought.envisage.ui.workbench.service_offers'

    # The plugin's unique identifier.
    id = 'TLoop.TLoop'

    # The plugin's name (suitable for displaying to the user).
    name = 'TLoop'

    # Services we contribute.
    service_offers = List(contributes_to=SERVICE_OFFERS)

    ######################################################################
    # Private methods.
    def _service_offers_default(self):
        """ Trait initializer. """
        ibvpy_service_offer = ServiceOffer(
            protocol='ibvpy.plugins.tloop_service.TLoopService',
            factory='ibvpy.plugins.tloop_service.TLoopService'
        )

        return [ibvpy_service_offer]
