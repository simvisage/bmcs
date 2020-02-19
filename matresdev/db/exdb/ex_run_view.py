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
#
# Created on Jan 19, 2010 by: rch

import os

from matplotlib.figure import \
    Figure
from pyface.api import \
    ImageResource, confirm, YES
from traits.api import \
    Instance, File, List, on_trait_change, Bool, \
    Event, Enum
from traitsui.api import \
    View, Item, ModelView, HSplit, VSplit, \
    Group, FileEditor
from traitsui.key_bindings import \
    KeyBinding, KeyBindings
from traitsui.menu import \
    Action, HelpAction, Menu, \
    MenuBar, ToolBar
from util.traits.editors.mpl_figure_editor import \
    MPLFigureEditor

from .ex_run import \
    ExRun
from matresdev.db.simdb.simdb import simdb


class ExRunView(ModelView):

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    filter = List()

    def _filter_default(self):
        return ['*.DAT;*.raw']

    # Overload the constructor to load the initial experiment.
    # The model of a ModelView cannot be None.
    #
    def __init__(self, **kw):
        super(ExRunView, self).__init__(**kw)
        self.load_run()
        self._reset_plot_template_list()
        self.redraw()

    # Register the ui window in order to be able to change its state
    #
    def init(self, ui_info):
        self._ui_info = ui_info

    #-------------------------------------------------------------------------
    # Model manipulation
    #-------------------------------------------------------------------------

    # Current experiment run
    #
    data_file = File

    def _data_file_default(self):
        # get the default form of the model
        #        ex_path = os.path.join( simdb.exdata_dir,
        #                                'tensile_tests',
        #                                'TT-9a',
        #                                'TT09-9a-V1.DAT' )
        ex_path = os.path.join(simdb.exdata_dir,
                               'plate_tests',
                               'PT-10a',
                               'PT11-10a.DAT')

        return ex_path

    # Reset the ex_run instance when the file was changed
    #
    @on_trait_change('data_file')
    def _reset_model(self, obj, name, old, new):
        print('RESET MODEL')
        model = self.model
        if model:
            # check if the model was changed and ask if it is to be saved
            #
            if model.unsaved:

                answer = confirm(self._ui_info.ui.control, 'Run changed, save it?',
                                 title='Save confirmation',
                                 cancel=False,
                                 default=YES)

                if answer == YES:
                    # ask whether the modified run should be saved
                    #
                    self.save_run()
                self.model.unsaved = False

        self.load_run()
        self._reset_plot_template_list()
        self.redraw()

    # The ex_run instance itself
    #
    model = Instance(ExRun)

    #-------------------------------------------------------------------------
    # Drawing
    #-------------------------------------------------------------------------

    # The variable controlling the redraw button. It is set True whenever
    # an input value of the experiment run was changed.
    #
    def set_changed(self):
        '''Callback to be registered with the model. It gets called whenever
        the change_event event of the ex_run has been triggered.
        '''
        self.redraw()

    unsaved = Bool(False)

    def listen_to_unsaved(self):
        self.unsaved = self.model.unsaved

    @on_trait_change('model')
    def _reset_model_listeners(self, obj, name, old_model, new_model):
        '''Delete the listeners of the old object reacting to the redraw tag
        of the experiment run. Bind the redraw listener to the newly
        attached model.
        '''
        if old_model:
            old_model.on_trait_change(
                self.set_changed, 'change_event', remove=True)
            old_model.on_trait_change(
                self.listen_to_unsaved, 'unsaved', remove=True)
        if new_model:
            new_model.on_trait_change(self.set_changed, 'change_event')
            new_model.on_trait_change(self.listen_to_unsaved, 'unsaved')

    @on_trait_change('plot_template')
    def redraw(self, ui_info=None):
        ''' Use the currently selected plot template to plot it in the Figure.
        '''
        print('*** replotting ***')

        # map the array dimensions to the plot axes
        #
        figure = self.figure
        axes = figure.gca()
        axes.clear()

        proc_name = self.model.ex_type.plot_templates[self.plot_template]
        plot_processor = getattr(self.model.ex_type, proc_name)
        plot_processor(axes)

        self.data_changed = True

    # -------------------------------------------------------------------------
    # Persistence management
    # -------------------------------------------------------------------------

    def load_run(self, ui_info=None):
        '''Read the run from the current data file into the model.
        '''
        self.model = ExRun(self.data_file)
        self.listen_to_unsaved()

    def save_run(self, ui_info=None):
        '''Save the model into associated pickle file.
        The source DAT file is unaffected.
        '''
        self.model.save_pickle()
        self.model.unsaved = False

    def reset_run(self, ui_info=None):
        '''Save the model into associated pickle file.
        The source DAT file is unaffected.
        '''
        answer = confirm(self._ui_info.ui.control, 'Really reset? Changes will be lost!',
                         title='Reset confirmation',
                         cancel=False,
                         default=YES)

        if answer == YES:
            # ask whether the modified run should be saved
            #
            self.load_run(ui_info)
            self.redraw()
            self.model.unsaved = False

    #-------------------------------------------------------------------
    # PLOT OBJECT
    #-------------------------------------------------------------------

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.add_axes([0.08, 0.13, 0.85, 0.74])
        return figure

    # event to trigger the replotting - used by the figure editor
    #
    data_changed = Event

    # selected plot template
    #
    plot_template = Enum(values='plot_template_list')

    def _plot_template_default(self):
        return self.model.ex_type.default_plot_template

    # list of availble plot templates
    # (gets extracted from the model whenever it's been changed)
    #
    plot_template_list = List

    def _reset_plot_template_list(self):
        '''Change the selection list of plot templates.

        This method is called upon every change of the model. This makes the viewing of
        different experiment types possible.
        '''
        self.plot_template_list = list(self.model.ex_type.plot_templates.keys())

    #-------------------------------------------------------------------------
    # UI specification
    #-------------------------------------------------------------------------

    key_bindings = Instance(KeyBindings)

    def _key_bindings_default(self):
        """ Trait initializer. """

        key_bindings = KeyBindings(
            KeyBinding(
                binding1='Ctrl-s',
                description='Save the run',
                method_name='save_run'
            ),

            KeyBinding(
                binding1='Ctrl-r',
                description='Plot the response',
                method_name='reset_run'
            )
        )

        return key_bindings

    def default_toolbar(self):
        return ToolBar(
            Action(name="Save",
                   tooltip='Save run',
                   enabled_when='unsaved',
                   image=ImageResource('save'),
                   action="save_run"),
            Action(name="Reset",
                   tooltip='Reset run',
                   enabled_when='unsaved',
                   image=ImageResource('reset'),
                   action="reset_run"),
            image_size=(22, 22),
            show_tool_names=False,
            show_divider=True,
            name='exrun_toolbar')

    def default_menubar(self):
        return MenuBar(Menu(Action(name="&Open",
                                   action="load_run"),
                            Action(name="&Save",
                                   action="save_run"),
                            Action(name="&Exit",
                                   action="exit"),
                            name="&File"),
                       Menu(Action(name="About PStudy",
                                   action="about_pstudy"),
                            HelpAction,
                            name="Help")
                       )

    def default_traits_view(self):

        return View(
            HSplit(
                VSplit(
                    Item('data_file@', editor=FileEditor(filter_name='filter'),
                         show_label=False),
                    Group(
                        Item('figure', editor=MPLFigureEditor(),
                             resizable=True, show_label=False),
                        id='simexdb.plot_sheet',
                        label='plot sheet',
                        dock='tab',
                    ),
                    Group(
                        Item('plot_template'),
                        columns=1,
                        label='plot parameters',
                        id='simexdb.plot_params',
                        dock='tab',
                    ),
                    id='simexdb.plot.vsplit',
                    dock='tab',
                ),
                VSplit(
                    Item('model@',
                         id='simexdb.run.split',
                         dock='tab',
                         resizable=True,
                         label='experiment run',
                         show_label=False),
                    id='simexdb.mode_plot_data.vsplit',
                    dock='tab',
                    scrollable=True
                ),
                id='simexdb.hsplit',
                dock='tab',
            ),
            key_bindings=self.key_bindings,
            menubar=self.default_menubar(),
            toolbar=self.default_toolbar(),
            resizable=True,
            title='Simvisage: experiment database browser',
            id='simexdb',
            dock='tab',
            buttons=['Ok'],
            height=1.0,
            width=1.0
        )

if __name__ == '__main__':

    #    ex_path = os.path.join( simdb.exdata_dir, 'bending_tests', 'ZiE_2011-06-08_BT-12c-6cm-0-TU', 'BT-12c-6cm-0-Tu-V4.raw' )

    ex_path = os.path.join(simdb.exdata_dir, 'tensile_tests',
                           'dog_bone', '2011-06-10_TT-12c-6cm-90-TU_ZiE',
                           'TT-12c-6cm-90-TU-V1.DAT')

#    ex_path = os.path.join( simdb.exdata_dir, 'plate_tests', 'PT-6a-ibac',
#                            'PTi-6a-woSF', 'PTi-6a-woSF-V1.DAT' )

#    ex_path = os.path.join( simdb.exdata_dir, 'plate_tests', 'PT-10a',
#                            'PT10-10a.DAT' )

    doe_reader = ExRunView(data_file=ex_path)
    doe_reader.configure_traits()
