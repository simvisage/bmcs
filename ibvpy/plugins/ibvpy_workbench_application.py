"""Mayavi specific workbench application.
"""
# Author: Prabhu Ramachandran <prabhu [at] aero . iitb . ac . in>
# Copyright (c) 2008, Enthought, Inc.
# License: BSD Style.

# Standard library imports.
from os.path import dirname

# Enthought library imports.
from envisage.ui.workbench.api import WorkbenchApplication
from pyface.api import AboutDialog, ImageResource, SplashScreen

# Local imports.
import mayavi.api
from mayavi.preferences.api import preference_manager

IMG_DIR = dirname(mayavi.api.__file__)

from pyface.message_dialog import MessageDialog


class AboutDialog(MessageDialog):
    title = 'About Simvisage.IBVPy'
    message = '''Authors:\nRostislav Chudoba,\nJakub Jerabek,\nAlexander Scholzen\n(C)2008
    '''


class IBVPyWorkbenchApplication(WorkbenchApplication):

    """ The mayavi application. """

    #### 'IApplication' interface #############################################

    # The application's globally unique Id.
    id = 'simvisage.ibvpy'

    #### 'WorkbenchApplication' interface #####################################

    # Branding information.
    #
    # The icon used on window title bars etc.
    icon = ImageResource('m2.ico', search_path=[IMG_DIR])

    # The name of the application (also used on window title bars etc).
    name = 'Simvisage.IBVPy'

    # Define an about dialog
    about_dialog = AboutDialog()

    ###########################################################################
    # 'WorkbenchApplication' interface.
    ###########################################################################

    def _about_dialog_default(self):
        """ Trait initializer. """

        about_dialog = AboutDialog(
            #            parent = self.workbench.active_window.control,
            #            image  = ImageResource('m2_about.jpg',
            #                                   search_path=[IMG_DIR]),
            additions=['Authors: Rostislav Chudoba',
                       'and Jakub Jerabek',
                       'and Alexander Scholzen'],
        )

        return about_dialog

    def _splash_screen_default(self):
        """ Trait initializer. """
        if preference_manager.root.show_splash_screen:
            splash_screen = SplashScreen(
                image=ImageResource('m2_about.jpg',
                                    search_path=[IMG_DIR]),
                show_log_messages=False,
            )
        else:
            splash_screen = None

        return splash_screen
