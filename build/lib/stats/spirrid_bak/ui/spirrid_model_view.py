#-------------------------------------------------------------------------------
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
# Created on Jan 6, 2011 by: rch

'''
Created on Sep 22, 2009

@author: rch
'''
from etsproxy.traits.api import HasTraits, Float, Property, cached_property, \
                                Instance, List, on_trait_change, Int, Tuple, Bool, \
                                DelegatesTo, Event, WeakRef, String, Constant, Trait, \
                                Button, Str, Enum

from etsproxy.traits.ui.api import \
    View, Item, Tabbed, VGroup, HGroup, Group, ModelView, HSplit, VSplit, Spring, TabularEditor, \
    TableEditor, Label

from etsproxy.traits.ui.table_column import \
    ObjectColumn, TableColumn

from etsproxy.traits.ui.extras.checkbox_column \
    import CheckboxColumn

from etsproxy.traits.ui.tabular_adapter \
    import TabularAdapter

from etsproxy.traits.ui.menu import OKButton

from util.traits.either_type import EitherType

from stats.pdistrib.pdistrib import PDistrib, IPDistrib

from numpy import array, linspace, frompyfunc, zeros, column_stack, \
                    log as ln, append, logspace, hstack, sign, trapz, \
                    arange

from stats.spirrid_bak.rf_filament import Filament
from stats.spirrid_bak import SPIRRID, RV, IRF
from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure

from stats.spirrid_bak import RFModelView

#----------------------------------------------------------------------------------
#                                     RIDVariable
#----------------------------------------------------------------------------------
class RIDVariable(HasTraits):
    """
    Association between a random variable and distribution.
    """
    spirrid = WeakRef

    rf = WeakRef

    n_int = Int(20, enter_set = True, auto_set = False,
                 desc = 'Number of integration points')
    def _n_int_changed(self):
        if self.pd:
            self.pd.n_segments = self.n_int

    # should this variable be randomized

    random = Bool(False, randomization_changed = True)
    def _random_changed(self):
        # get the default distribution
        if self.random:
            self.spirrid.rv_dict[ self.varname ] = RV(pd = self.pd, name = self.varname, n_int = self.n_int)
        else:
            del self.spirrid.rv_dict[ self.varname ]

    # name of the random variable (within the response function)
    #
    varname = String

    trait_value = Float

    pd = Property(Instance(IPDistrib), depends_on = 'random')
    @cached_property
    def _get_pd(self):
        if self.random:
            tr = self.rf.trait(self.varname)
            pd = PDistrib(distr_choice = tr.distr[0], n_segments = self.n_int)
            trait = self.rf.trait(self.varname)

            # get the distribution parameters from the metadata
            #
            distr_params = {'scale' : trait.scale, 'loc' : trait.loc, 'shape' : trait.shape }
            dparams = {}
            for key, val in list(distr_params.items()):
                if val:
                    dparams[key] = val

            pd.distr_type.set(**dparams)
            return pd
        else:
            return None

    value = Property
    def _get_value(self):
        if self.random:
            return ''
        else:
            return '%g' % self.trait_value

    # --------------------------------------------

    # default view specification
    def default_traits_view(self):
        return View(HGroup(
                                 Item('random'),
                                 Item('n_int', visible_when = 'random', label = 'NIP',
                                        ),
                                 Spring(),
                                 show_border = True,
                                 label = 'Variable name: %s' % self.varname
                                 ),
                    Item('pd@', show_label = False),
                    resizable = True,
                    id = 'rid_variable',
                    height = 800)

class PDColumn(ObjectColumn):
    name = 'value'

    def get_image(self, object):
        if object.pd:
            return object.pd.icon
        else:
            return self.image


rv_list_editor = TableEditor(
                    columns = [ ObjectColumn(name = 'varname', label = 'Name',
                                                editable = False,
                                                horizontal_alignment = 'center'),
                                  CheckboxColumn(name = 'random', label = 'Random',
                                                editable = True,
                                                horizontal_alignment = 'center'),
                                  PDColumn(label = 'Value',
                                                editable = False,
                                                horizontal_alignment = 'center'),
                                  ObjectColumn(name = 'n_int', label = 'NIP',
                                                editable = True,
                                                format = '%d',
                                                horizontal_alignment = 'center'),
 ],
                    selection_mode = 'row',
                    selected = 'object.selected_var',
                    deletable = False,
                    editable = False,
                    show_toolbar = True,
                    auto_add = False,
                    configurable = False,
                    sortable = False,
                    reorderable = False,
                    sort_model = False,
                    orientation = 'vertical',
                    auto_size = True,
        )

class SPIRRIDModelView(ModelView):
    '''
    Size effect depending on the yarn length
    '''
    model = Instance(SPIRRID)
    def _model_changed(self):
        self.model.rf = self.rf

    rf_values = List(IRF)
    def _rf_values_default(self):
        return [ Filament() ]

    rf = Enum(values = 'rf_values')
    def _rf_default(self):
        return self.rf_values[0]
    def _rf_changed(self):
        # reset the rf in the spirrid model and in the rf_modelview
        self.model.rf = self.rf
        self.rf_model_view = RFModelView(model = self.rf)
        # actually, the view should be reusable but the model switch
        # did not work for whatever reason
        # the binding of the view generated by edit_traits does not 
        # react to the change in the 'model' attribute.
        #
        # Remember - we are implementing a handler here
        # that has an associated view.
        #
        # self.rf.model_view.model = self.rf

    rf_model_view = Instance(RFModelView)
    def _rf_model_view_default(self):
        return RFModelView(model = self.rf)


    rv_list = Property(List(RIDVariable), depends_on = 'rf')
    @cached_property
    def _get_rv_list(self):
        return [ RIDVariable(spirrid = self.model, rf = self.rf,
                              varname = nm, trait_value = st)
                 for nm, st in zip(self.rf.param_keys, self.rf.param_values) ]

    selected_var = Instance(RIDVariable)
    def _selected_var_default(self):
        return self.rv_list[0]

    run = Button(desc = 'Run the computation')
    def _run_fired(self):
        self._redraw()

    sample = Button(desc = 'Show samples')
    def _sample_fired(self):
        n_samples = 50

        self.model.set(
                    min_eps = 0.00, max_eps = self.max_eps, n_eps = self.n_eps,
                    )

        # get the parameter combinations for plotting
        rvs_theta_arr = self.model.get_rvs_theta_arr(n_samples)

        eps_arr = self.model.eps_arr

        figure = self.figure
        axes = figure.gca()

        for theta_arr in rvs_theta_arr.T:
            q_arr = self.rf(eps_arr, *theta_arr)
            axes.plot(eps_arr, q_arr, color = 'grey')

        self.data_changed = True

    run_legend = Str('',
                     desc = 'Legend to be added to the plot of the results')

    clear = Button
    def _clear_fired(self):
        axes = self.figure.axes[0]
        axes.clear()
        self.data_changed = True

    min_eps = Float(0.0,
                     desc = 'minimum value of the control variable')

    max_eps = Float(1.0,
                     desc = 'maximum value of the control variable')

    n_eps = Int(100,
                 desc = 'resolution of the control variable')

    label_eps = Str('epsilon',
                    desc = 'label of the horizontal axis')

    label_sig = Str('sigma',
                    desc = 'label of the vertical axis')

    figure = Instance(Figure)
    def _figure_default(self):
        figure = Figure(facecolor = 'white')
        #figure.add_axes( [0.08, 0.13, 0.85, 0.74] )
        return figure

    data_changed = Event(True)

    def _redraw(self):

        figure = self.figure
        axes = figure.gca()

        self.model.set(
                    min_eps = 0.00, max_eps = self.max_eps, n_eps = self.n_eps,
                )

        mc = self.model.mean_curve

        axes.plot(mc.xdata, mc.ydata,
                   linewidth = 2, label = self.run_legend)

        axes.set_xlabel(self.label_eps)
        axes.set_ylabel(self.label_sig)
        axes.legend(loc = 'best')

        self.data_changed = True

    traits_view_tabbed = View(
                              VGroup(
                              HGroup(
                                    Item('run_legend', resizable = False, label = 'Run label',
                                          width = 80, springy = False),
                                    Item('run', show_label = False, resizable = False),
                                    Item('sample', show_label = False, resizable = False),
                                    Item('clear', show_label = False,
                                      resizable = False, springy = False)
                                ),
                               Tabbed(
                               VGroup(
                                     Item('rf', show_label = False),
                                     Item('rf_model_view@', show_label = False, resizable = True),
                                     label = 'Deterministic model',
                                     id = 'spirrid.tview.model',
                                     ),
                                Group(
                                    Item('rv_list', editor = rv_list_editor, show_label = False),
                                    id = 'spirrid.tview.randomization.rv',
                                    label = 'Model variables',
                                ),
                                Group(
                                    Item('selected_var@', show_label = False, resizable = True),
                                    id = 'spirrid.tview.randomization.distr',
                                    label = 'Distribution',
                                ),
                                VGroup(
                                       Item('model.cached_dG' , label = 'Cached weight factors',
                                             resizable = False,
                                             springy = False),
                                       Item('model.compiled_QdG_loop' , label = 'Compiled loop over the integration product',
                                             springy = False),
                                       Item('model.compiled_eps_loop' ,
                                             enabled_when = 'model.compiled_QdG_loop',
                                             label = 'Compiled loop over the control variable',
                                             springy = False),
                                        scrollable = True,
                                       label = 'Execution configuration',
                                       id = 'spirrid.tview.exec_params',
                                       dock = 'tab',
                                     ),
                                VGroup(
                                          HGroup(
                                                 Item('min_eps' , label = 'Min',
                                                       springy = False, resizable = False),
                                                 Item('max_eps' , label = 'Max',
                                                       springy = False, resizable = False),
                                                 Item('n_eps' , label = 'N',
                                                       springy = False, resizable = False),
                                                 label = 'Simulation range',
                                                 show_border = True
                                                 ),
                                          VGroup(
                                                 Item('label_eps' , label = 'x', resizable = False,
                                                 springy = False),
                                                 Item('label_sig' , label = 'y', resizable = False,
                                                 springy = False),
                                                 label = 'Axes labels',
                                                 show_border = True,
                                                 scrollable = True,
                                                 ),
                                           label = 'Execution control',
                                           id = 'spirrid.tview.view_params',
                                           dock = 'tab',
                                 ),
                                VGroup(
                                        Item('figure', editor = MPLFigureEditor(),
                                             resizable = True, show_label = False),
                                        label = 'Plot sheet',
                                        id = 'spirrid.tview.figure_window',
                                        dock = 'tab',
                                        scrollable = True,
                                ),
                                scrollable = True,
                                id = 'spirrid.tview.tabs',
                                dock = 'tab',
                        ),
                        ),
                        title = 'SPIRRID',
                        id = 'spirrid.viewmodel',
                        dock = 'tab',
                        resizable = True,
                        height = 1.0, width = 1.0
                        )


def run():

    sv = SPIRRIDModelView(model = SPIRRID(),
                           rf_values = [ Filament() ])
    sv.configure_traits(view = 'traits_view_tabbed')

if __name__ == '__main__':
    run()
