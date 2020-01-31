#!/usr/bin/env python
""" The entry point for an Envisage application. """

# Standard library imports.
import sys
import os.path
import logging

# Enthought library imports.
#from enthought.mayavi.plugins.app import get_plugins, setup_logger
from enthought.mayavi.plugins.app import setup_logger
from enthought.traits.api import List, Instance
from enthought.envisage.api import Plugin, ServiceOffer, ExtensionPoint
from enthought.pyface.workbench.api import Perspective, PerspectiveItem

###############################################################################
# `TStepperPlugin` class.
###############################################################################
class TStepperUIPlugin(Plugin):

    # Extension points we contribute to.
    VIEWS             = 'enthought.envisage.ui.workbench.views'

    # The plugin's unique identifier.
    id = 'tstepper_service.tstepper_service'

    # The plugin's name (suitable for displaying to the user).
    name = 'TStepper Manager'

    # Views.
    views = List(contributes_to=VIEWS)

    ######################################################################
    # Private methods.
    def _views_default(self):
        """ Trait initializer. """
        return [self._tstepper_service_view_factory]

    def _tstepper_service_view_factory(self, window, **traits):
        """ Factory method for tstepper_service views. """
        from enthought.pyface.workbench.traits_ui_view import \
                TraitsUIView

        tstepper_service = self._get_tstepper_service(window)
        tui_engine_view = TraitsUIView(obj=tstepper_service,
                                       id='ibvpy.plugins.tstepper_service.tstepper_service',
                                       name='Time stepper',
                                       window=window,
                                       position='left',
                                       **traits
                                       )
        return tui_engine_view

    def _get_tstepper_service(self,window):
        """Return the tstepper_service service."""
        return window.get_service('ibvpy.plugins.tstepper_service.TStepperService' )
