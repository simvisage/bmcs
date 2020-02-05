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


from matplotlib.figure import Figure
from numpy import \
    zeros
from pyface.api import ImageResource
from traits.api import Property, cached_property, \
    Instance, List, Int, \
    DelegatesTo, Event, Str, Array, Any
from traitsui.api import \
    View, Item, VGroup, ModelView, HSplit, VSplit, \
    EnumEditor
from traitsui.menu import Action, \
    ToolBar
from traitsui.menu import \
    OKButton
from util.traits.editors.mpl_figure_editor import MPLFigureEditor

from .i_sim_array import \
    ISimArray
from .sim_factor import \
    SimFactor
from .sim_output import \
    SimOut
from .sim_todo import \
    ToDo


class SimArrayView(ModelView):
    '''
    View into the parametric space constructed over the model.

    The is associated with the PStudySpace instance covering the
    factor ranges using an n-dimensional array.

    The view is responsible for transferring the response values
    into 2D and 3D plots. Depending on the current view specification
    it also initiates the calculation of response values in the 
    currently viewed subspace of the study. 
    '''

    model = Instance(ISimArray)

    #---------------------------------------------------------------
    # PARAMETER RANGE SPECIFICATION
    #-------------------------------------------------------------------
    factor_dict = DelegatesTo('model')

    # alphabetically ordered names of factors
    #
    factor_names = DelegatesTo('model')

    # alphabetically ordered list of factors
    #
    factor_list = DelegatesTo('model')

    #---------------------------------------------------------------
    # X-PARAMETER RANGE SPECIFICATION
    #-------------------------------------------------------------------
    # Selected factor name for evalution along the X-axis
    #
    x_factor_name = Str(factors_modified=True)

    def _x_factor_name_default(self):
        return self.factor_names[0]

    def _x_factor_name_changed(self):
        if self.x_factor_name == self.y_factor_name:
            self.y_factor_name = '-'
        if self.x_factor_name == self.other_factor_name:
            self.other_factor_name = '-'
        self.x_factor = self.factor_dict[self.x_factor_name]

    x_factor = Instance(SimFactor)

    def _x_factor_default(self):
        return self.factor_dict[self.factor_names[0]]

    # index of the currently selected variable
    x_factor_idx = Property()

    def _get_x_factor_idx(self):
        return self.factor_names.index(self.x_factor_name)
    #---------------------------------------------------------------
    # Y-PARAMETER RANGE SPECIFICATION
    #-------------------------------------------------------------------
    y_factor_names = Property(depends_on='x_factor_name')

    @cached_property
    def _get_y_factor_names(self):
        current_x_factor = self.x_factor_name
        current_x_factor_idx = self.factor_names.index(current_x_factor)
        y_factor_names = self.factor_names[:current_x_factor_idx] + \
            self.factor_names[current_x_factor_idx + 1:]
        return ['-'] + y_factor_names
    #
    # Selected factor name for evalution of multiple lines
    #
    y_factor_name = Str('-', factors_modified=True)

    def _y_factor_name_changed(self):
        if self.y_factor_name == self.other_factor_name:
            self.other_factor_name = '-'
        if self.y_factor_name == '-':
            self.y_factor = None
        else:
            self.y_factor = self.factor_dict[self.y_factor_name]
    y_factor = Instance(SimFactor)

    y_factor_idx = Property()

    def _get_y_factor_idx(self):
        return self.factor_names.index(self.y_factor_name)

    #------------------------------------------------------------------
    # OTHER PARAM LEVELS
    #------------------------------------------------------------------
    other_factor_names = Property(depends_on='x_factor_name, y_factor_name')

    @cached_property
    def _get_other_factor_names(self):
        x_factor_idx = self.factor_names.index(self.x_factor_name)

        y_factor_idx = x_factor_idx
        if self.y_factor_name != '-':
            y_factor_idx = self.factor_names.index(self.y_factor_name)

        ignore_idx = [x_factor_idx, y_factor_idx]
        ignore_idx.sort()

        other_factor_names = self.factor_names[:ignore_idx[0] ] + \
            self.factor_names[ignore_idx[0] + 1: ignore_idx[1]] + \
            self.factor_names[ignore_idx[1] + 1:]

        return ['-'] + other_factor_names

    #
    # Selected factor name for evalution of multiple lines
    #
    other_factor_name = Str('-', factors_modified=True)

    def _other_factor_name_changed(self):
        if self.other_factor_name == '-':
            self.other_factor_levels = []
        else:
            levels = self.factor_dict[self.other_factor_name].level_list
            self.other_factor_levels = list(levels)
            other_factor_idx = self.factor_names.index(self.other_factor_name)
            other_factor_level_idx = self.frozen_factor_levels[
                other_factor_idx]
            self.other_factor_level = self.other_factor_levels[
                other_factor_level_idx]

    other_factor_levels = List
    other_factor_level = Any

    def _other_factor_level_changed(self):
        level_idx = self.other_factor_levels.index(self.other_factor_level)
        other_factor_idx = self.factor_names.index(self.other_factor_name)
        self.frozen_factor_levels[other_factor_idx] = level_idx

    frozen_factor_levels = Array

    def _frozen_factor_levels_default(self):
        return zeros(self.model.n_factors, dtype='int_')
    #---------------------------------------------------------------
    # OUTPUT ARRAY SPECIFICATION
    #-------------------------------------------------------------------
    outputs = DelegatesTo('model')

    # extract the available names
    output_names = DelegatesTo('model')

    # active selection to be plotted
    output_name = Str

    def _output_name_default(self):
        return self.output_names[0]
    output_idx = Property(Int, depends_on='output_name')

    def _get_output_idx(self):
        return self.output_names.index(self.output_name)

    def _output_name_changed(self):
        self.output = self.outputs[self.output_idx]
    output = Instance(SimOut)

    def _output_default(self):
        return self.outputs[0]

    #---------------------------------------------------------------
    # PLOT OBJECT
    #-------------------------------------------------------------------
    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.add_axes([0.12, 0.13, 0.85, 0.74])
        return figure

    #-------------------------------------------------------------------------
    # Public Controller interface
    #-------------------------------------------------------------------------
    def start_study(self, ui_info):
        self._start_study()

    def stop_study(self, ui_info):
        todo = ToDo()
        todo.configure_traits(kind='modal')

    def _start_study(self):

        # identify the runs to be performed
        # use slices along the varied factors
        # to obtain the indices of the values.

        # get the sliced dimensions
        #
        factor_levels = [level for level in self.frozen_factor_levels]
        factor_levels[self.x_factor_idx] = slice(None)
        if self.y_factor_name != '-':
            factor_levels[self.y_factor_idx] = slice(None)

        factor_slices = tuple(factor_levels)

        # get the response value for the given factor slices
        #
        output_array = self.model[factor_slices]

        # map the array dimensions to the plot axes
        #
        figure = self.figure

        axes = figure.axes[0]
        axes.clear()

        x_levels = self.x_factor.level_list

        if self.y_factor_name == '-':

            axes.plot(x_levels, output_array[:, self.output_idx]
                      # color = c, linewidth = w, linestyle = s
                      )
        else:
            y_levels = self.y_factor.level_list
            for i_y, y_level in enumerate(y_levels):

                index = x_levels

                # The dimensions of the returned array are given
                # by the index of the factors within the pstudy
                # In other words, the subspace to be plotted has
                # the same order of factors as the original space.
                # The remapping of the axes must therefore respect
                # this order and take y data from the first dimension
                # if y_factor_idx is lower than y_factor_idx
                #
                if self.y_factor_idx > self.x_factor_idx:
                    values = output_array[:, i_y, self.output_idx]
                else:
                    values = output_array[i_y, :, self.output_idx]

                axes.plot(index, values
                          # color = c, linewidth = w, linestyle = s
                          )
            legend = [str(level) for level in y_levels]
            axes.legend(legend, loc='best')

        axes.set_xlabel('%s [%s]' % (self.x_factor_name, self.x_factor.unit),
                        weight='semibold')
        axes.set_ylabel('%s [%s]' % (self.output_name, self.output.unit),
                        weight='semibold')

#        axes.set_title( 'strength size effect',\
#                        size = 'large', color = 'black',\
#                        weight = 'bold', position = (.5,1.03))
        axes.set_axis_bgcolor(color='white')
#        axes.ticklabel_format(scilimits = (-3.,4.))
        axes.grid(color='gray', linestyle='--', linewidth=0.1, alpha=0.4)
#        axes.legend(( legend ), loc = 'best')
#        axes.set_xscale('log' ) #, subsx = [0, 1, 2, 3 ] )
#        axes.set_yscale('log' ) # , subsy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] )
        # axes.set_xlim(10.e-4)

        self.data_changed = True

    data_changed = Event

    traits_view = View(
        HSplit(
            VGroup(
                VGroup(
                    Item('x_factor_name',
                         editor=EnumEditor(name='factor_names'),
                         label='horizontal axis'),
                    Item('y_factor_name',
                         editor=EnumEditor(name='y_factor_names'),
                         label='depth axis'),
                    Item('other_factor_name',
                         editor=EnumEditor(name='other_factor_names'),
                         label='other factors'),
                    Item('other_factor_level',
                         editor=EnumEditor(name='other_factor_levels'),
                         label='viewed level'),
                    label='viewed subspace',
                    id='sim_pstudy.viewmodel.factor.subspace',
                    dock='tab',
                    scrollable=True,
                ),
                VGroup(
                    Item('output_name',
                         editor=EnumEditor(name='output_names'),
                         show_label=False),
                    Item('output@',
                         show_label=False,
                         springy=True),
                    label='vertical axis',
                    id='sim_psrudy.viewmodel.control',
                    dock='tab',
                ),
                id='sim_pstudy.viewmodel.left',
                label='studied factors',
                layout='normal',
                dock='tab',
            ),
            VSplit(
                VGroup(
                    Item('figure',  editor=MPLFigureEditor(),
                         resizable=True, show_label=False),
                    label='plot sheet',
                    id='sim_pstudy.viewmode.figure_window',
                    dock='tab',
                ),
                id='sim_pstudy.viewmodel.right',
            ),
            id='sim_pstudy.viewmodel.splitter',
            #group_theme = '@G',
            #item_theme  = '@B0B',
            #label_theme = '@BEA',
        ),
        toolbar=ToolBar(
            Action(name="Run",
                   tooltip='Start computation',
                   image=ImageResource('kt-start'),
                   action="start_study"),
            Action(name="Pause",
                   tooltip='Pause computation',
                   image=ImageResource('kt-pause'),
                   action="pause_study"),
            Action(name="Stop",
                   tooltip='Stop computation',
                   image=ImageResource('kt-stop'),
                   action="stop_study"),
            image_size=(32, 32),
            show_tool_names=False,
            show_divider=True,
            name='view_toolbar'),
        title='SimVisage Component: Parametric Studies',
        id='sim_pstudy.viewmodel',
        dock='tab',
        resizable=True,
        height=0.8, width=0.8,
        buttons=[OKButton])


def run():
    from .sim_model import SimModel
    from .sim_array import SimArray

    pstudy = SimArray(sim_model=SimModel())
    pstudy_view = SimArrayView(model=pstudy)
    pstudy_view.configure_traits()

if __name__ == '__main__':
    run()
