#!/usr/bin/env python
""" The entry point for an Envisage application. """

# Enthought library imports.
from traits.api import List
from envisage.api import Plugin

###############################################################################
# `SDomainPlugin` class.
###############################################################################


class SDomainUIPlugin(Plugin):

    # Extension points we contribute to.
    VIEWS = 'enthought.envisage.ui.workbench.views'

    # The plugin's unique identifier.
    id = 'sdomain_service.sdomain_service'

    # The plugin's name (suitable for displaying to the user).
    name = 'Spatial domain'

    # Views.
    views = List(contributes_to=VIEWS)

    ######################################################################
    # Private methods.
    def _views_default(self):
        """ Trait initializer. """
        return [self._sdomain_service_view_factory]

    def _sdomain_service_view_factory(self, window, **traits):
        """ Factory method for sdomain_service views. """
        from pyface.workbench.traits_ui_view import \
            TraitsUIView

        sdomain_service = self._get_sdomain_service(window)
        tui_engine_view = TraitsUIView(obj=sdomain_service,
                                       id='ibvpy.plugins.sdomain_service.sdomain_service',
                                       name='Spatial domain',
                                       window=window,
                                       position='left',
                                       **traits
                                       )
        return tui_engine_view

    def _get_sdomain_service(self, window):
        """Return the sdomain_service service."""
        return window.get_service('ibvpy.plugins.sdomain_service.SDomainService')
