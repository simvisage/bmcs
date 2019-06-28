'''
Created on Apr 24, 2019

@author: rch

Changes since Friday June 7 2019 - BMCS coding workshop
    * refresh of MPLFigureEditor - the curve gets reploted
      when the model object changes ist own event - data_changed.
      If the model object has several Matplotlib instances
      it must provide the trait event as its attribute 
      issue the event if the replot should be done
      
      self.data_changed = True

'''
import os
import string

from matplotlib.figure import Figure
from pyface.api import FileDialog, MessageDialog
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from util.traits.editors import MPLFigureEditor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traits.api as tr
import traitsui.api as ui
from traitsui.tabular_adapter \
    import TabularAdapter

from .hcff_filters.hcff_filter import HCFFilter, HCFFChild, HCFFParent
from .hcff_import_manager import FileImportManager


class DataArrayAdapter(TabularAdapter):

    columns = tr.Property

    def _get_columns(self):
        columns = self.object.columns
        return [(name, idx) for idx, name in enumerate(columns)]

    font = 'Courier 10'
    alignment = 'right'
    format = '%5.4f'  # '%g'
    even_bg_color = tr.Color(0xE0E0FF)
    width = tr.Float(80)


class HCFFRoot(HCFFParent):

    name = tr.Str('root')

    import_manager = tr.Instance(FileImportManager, ())

    output_table = tr.Property(tr.Dict,
                               depends_on='+inputs')

    def _get_output_table(self):
        return {
            'first': np.linspace(0, 100, 20),
            'second': np.sin(np.linspace(0, 100, 20))
        }

    tree_view = ui.View(
        ui.VGroup(
            ui.Item('import_manager', style='custom', show_label=False)
        )
    )

    traits_view = tree_view


tree_editor = ui.TreeEditor(
    orientation='vertical',
    nodes=[
        # The first node specified is the top level one
        ui.TreeNode(node_for=[HCFFRoot],
                    auto_open=True,
                    children='filters',
                    label='=HCF',
                    #                    view=ui.View()  # Empty view
                    ),
        ui.TreeNode(node_for=[HCFFilter],
                    auto_open=True,
                    children='filters',
                    label='name',
                    #                    view=ui.View()
                    ),
    ],
    selected='selected_filter',
)


class HCFF(tr.HasStrictTraits):
    '''High-Cycle Fatigue Filter
    '''

    hcf = tr.Instance(HCFFRoot)

    def _hcf_default(self):
        return HCFFRoot(import_manager=FileImportManager())

    figure = tr.Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    selected_filter = tr.WeakRef(HCFFilter)
    data_changed = tr.Event

    def _selected_filter_changed(self):
        print('replotting')
        ax = self.figure.add_subplot(1, 1, 1)
        data = self.selected_filter.output_table
        y = data['second']
        x = data['first']
#        self.selected_filter.plot(ax)
        ax.plot(x, y)
        self.data_changed = True

    traits_view = ui.View(
        ui.HSplit(
            ui.Item(name='hcf',
                    editor=tree_editor,
                    show_label=False,
                    width=0.3
                    ),
            ui.VSplit(
                ui.UItem('figure', editor=MPLFigureEditor(),
                         resizable=True,
                         springy=True,
                         label='2d plots'
                         ),
            ),


        ),
        title='HCF Filter',
        resizable=True,
        width=0.6,
        height=0.6
    )


if __name__ == '__main__':
    #     hcff = HCFF(file_csv='C:\\Users\\hspartali\\Desktop\\BeamEnd_Results')
    hcff = HCFF()

    cut_initial = HCFFilter(name='cut')
    hcff.hcf.add_filter(cut_initial)

    directly_make_average = HCFFilter(name='average1')
    hcff.hcf.add_filter(directly_make_average)

    hcff.selected_filter = cut_initial

    custom_filter = HCFFilter(name='custom')
    cut_initial.add_filter(custom_filter)

    custom_filter.add_filter(HCFFilter(name='average'))
    cut_initial.add_filter(HCFFilter(name='custom differently'))


#    plot_options = [PlotOptions()]
    hcff.configure_traits()
