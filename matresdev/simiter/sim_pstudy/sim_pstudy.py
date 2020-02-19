#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
# ve_study
# Created on Feb 1, 2010 by: rch

""" Application window. """


from math import exp, e, sqrt, log, pi
import os
import pickle
import string

from matplotlib.figure import Figure
from numpy import array, linspace, frompyfunc, zeros, column_stack, \
    log as ln, append, logspace, hstack, sign, trapz, mgrid, c_, \
    zeros, arange, ix_
from pyface.api import ImageResource
from pyface.api import confirm, error, information, warning, YES, NO, CANCEL
from scipy.optimize import brentq, newton
from scipy.special import erf, gamma
from scipy.stats import norm, weibull_min, uniform
from traits.api import \
    HasTraits, Float, Property, cached_property, \
    Instance, List, on_trait_change, Int, Tuple, Bool, \
    DelegatesTo, Event, Str, Button, Dict, Array, Any, \
    implements, File
from traitsui.api import \
    View, Item, Tabbed, VGroup, HGroup, ModelView, HSplit, VSplit, \
    CheckListEditor, EnumEditor, TableEditor, TabularEditor, Handler
from traitsui.file_dialog  \
    import open_file, save_file, FileInfo, TextInfo
from traitsui.menu import Action, CloseAction, HelpAction, Menu, \
    MenuBar, NoButtons, Separator, ToolBar
from traitsui.menu import OKButton
from traitsui.tabular_adapter \
    import TabularAdapter
from util.traits.editors.mpl_figure_editor import MPLFigureEditor

from .i_sim_array import ISimArray
from .i_sim_model import ISimModel
import os.path as path
from .sim_array import SimArray
from .sim_array_view import SimArrayView
from .sim_factor import \
    SimFactor, SimFloatFactor, SimIntFactor, SimEnumFactor
from .sim_output import SimOut
from .sim_todo import ToDo


#os.environ['ETS_TOOLKIT'] = 'qt4'
#
TITLESTRING = 'simvisage.sim_array'


class SimPStudyController(Handler):

    def init(self, ui_info):
        '''Set the title string
        '''
        self._set_title_string(ui_info, ignore_dirty=True)
    #-------------------------------------------------------------------------
    # Public Controller interface
    #-------------------------------------------------------------------------

    def new_study(self, ui_info):

        if ui_info.object.dirty:

            # discard / save dialog

            answer = confirm(ui_info.ui.control, 'Study modified. Save it?',
                             title='New study',
                             cancel=False,
                             default=YES)

            if answer == YES:
                self.save_study(ui_info)

        ui_info.object.file_path = ''
        ui_info.object.new()
        self._set_title_string(ui_info)

    def exit_study(self, ui_info):

        if ui_info.object.dirty:

            # discard / save dialog
            answer = confirm(ui_info.ui.control, 'Save study before exiting?',
                             title='Close study',
                             cancel=True,
                             default=YES)

            if answer == YES:
                self.save_study(ui_info)
                self._on_close(ui_info)
                return True
            elif answer == NO:
                self._on_close(ui_info)
                return True
            else:
                return False
        else:
            self._on_close(ui_info)
            return True

#    def close(self, ui_info, is_ok ):
#        is_ok = self.exit_study( ui_info )
#        print 'IS OK', is_ok
#        super( SimPStudyController, self ).close( ui_info, is_ok )
#        self._on_close( ui_info )

    def open_study(self, ui_info):

        if ui_info.object.dirty:
            # discard / save dialog

            answer = confirm(ui_info.ui.control, 'Study modified. Save it?',
                             title='Open study',
                             cancel=True,
                             default=YES)

            if answer == YES:
                self.save_study(ui_info)
            elif answer == CANCEL:
                return

        file_name = open_file(filter=['*.pst'],
                              extensions=[FileInfo(), TextInfo()])
        if file_name != '':
            ui_info.object.load(file_name)
            ui_info.object.file_path = file_name
            self._set_title_string(ui_info)

    def save_study(self, ui_info):

        if ui_info.object.file_path == '':
            file_name = save_file(filter=['*.pst'],
                                  extensions=[FileInfo(), TextInfo()])
            if file_name == '':
                return
        else:
            file_name = ui_info.object.file_path

        ui_info.object.save(file_name)

    def save_study_as(self, ui_info):

        file_name = save_file(filter=['*.pst'],
                              extensions=[FileInfo(), TextInfo()])
        if file_name != '':
            ui_info.object.save(file_name)
            ui_info.object.file_path = file_name
            self._set_title_string(ui_info)

    def new_view(self, ui_info):
        new_view = SimArrayView(model=ui_info.object.sim_array)
        new_view.configure_traits()  # kind = 'livemodal' )

    def exit_pstudy(self, ui_info):
        '''Close all views and check if everything was saved'''
        todo = ToDo()
        todo.configure_traits(kind='modal')

    def clear_cache(self, ui_info):
        ui_info.object.clear_cache()

    def about_pstudy(self, ui_info):
        todo = ToDo()
        todo.configure_traits(kind='modal')

    def _set_title_string(self, ui_info, ignore_dirty=False):

        if ui_info.object.dirty and not ignore_dirty:
            modified = '(modified)'
        else:
            modified = ''
        if ui_info.object.file_path == '':
            filename = '<unnamed>'
        else:
            filename = ui_info.object.file_path

        title_string = '%s: %s %s' % (TITLESTRING, '<unnamed>', modified)
        ui_info.ui.title = title_string


class SimPStudy(HasTraits):
    """ The main application window. """

    def __init__(self, **kw):
        super(SimPStudy, self).__init__(**kw)

        # The initialization should not be considered dirty
        # therefore set the flag to indicate unsaved study to false
        #
        self.dirty = False

    sim_array = Instance(SimArray)

    def _sim_array_default(self):
        return SimArray()

    sim_model = Property()

    def _set_sim_model(self, value):
        self.sim_array.sim_model = value

    def __getitem__(self, factor_slices):
        '''Direct access to the sim_array.
        '''
        return self.sim_array[factor_slices]

    #---------------------------------------------------------------
    # PERSISTENCY
    #-------------------------------------------------------------------

    file_base_name = Property()

    def _get_file_base_name(self):
        return self.sim_model.__class__.__name__

    file_path = Str('')

    dirty = False

    @on_trait_change('sim_array.changed')
    def _set_dirty(self):
        self.dirty = True

    def new(self):
        sim_model = self.sim_array.sim_model
        self.sim_array = SimArray(sim_model=sim_model)
        self.dirty = False

    def load(self, file_name):
        file = open(file_name, 'r')
        self.sim_array = pickle.load(file)
        file.close()
        self.dirty = False

    def save(self, file_name):
        file = open(file_name, 'w')
        pickle.dump(self.sim_array, file)
        file.close()
        self.dirty = False

    def clear_cache(self):
        self.sim_array.clear_cache()

    toolbar = ToolBar(
        Action(name="New Study",
               tooltip='Create a new study',
               image=ImageResource('New-32'),
               action="new_study"),
        Action(name="Open Study",
               tooltip='Open a study',
               image=ImageResource('fileopen-32'),
               action="open_study"),
        Action(name="Save Study",
               tooltip='Save study',
               image=ImageResource('save'),
               action="save_study"),
        Action(name="New View",
               tooltip='Create new view',
               image=ImageResource('new_view'),
               action="new_view"),
        Action(name="Clear Cache",
               tooltip='Reset cache',
               image=ImageResource('reset'),
               action="clear_cache"),
        image_size=(22, 22),
        show_tool_names=False,
        show_divider=True,
        name='study_toolbar')

    menubar = MenuBar(Menu(Action(name="&New",
                                  action="new_study"),
                           Action(name="&Open",
                                  action="open_study"),
                           Action(name="&Save",
                                  action="save_study"),
                           Action(name="Save &As",
                                  action="save_study_as"),
                           Action(name="&Exit",
                                  action="exit_study"),
                           name="&File"),
                      Menu(Action(name="&New View",
                                  action="new_view"),
                           name="&View"),
                      Menu(Action(name="&Clear Cache",
                                  action="clear_cache"),
                           name="&Data"),
                      Menu(Action(name="About PStudy",
                                  action="about_pstudy"),
                           HelpAction,
                           name="Help")
                      )

    view = View(
        Item('sim_array@', show_label=False),
        id='simvisage.simiter.pstudy',
        dock='tab',
        menubar=menubar,
        toolbar=toolbar,
        resizable=True,
        width=0.8,
        height=0.8,
        title='SimVisage: Parametric Study',
        handler=SimPStudyController,
    )


def run():

    from .sim_model import SimModel
    pstudy_app = SimPStudy(sim_model=SimModel())
    pstudy_app.configure_traits(kind='live')

if __name__ == '__main__':
    run()
