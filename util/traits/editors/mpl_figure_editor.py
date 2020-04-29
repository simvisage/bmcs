

from traits.etsconfig.api import ETSConfig

if ETSConfig.toolkit == 'wx':
    from .mpl_figure_editor_wx import MPLFigureEditor
elif ETSConfig.toolkit == 'qt4':
    from .mpl_figure_editor_qt import MPLFigureEditor
else:
    from .mpl_figure_editor_void import MPLFigureEditor
#     raise ImportError("MPLEditor for %s toolkit not available" %
#                       ETSConfig.toolkit)
