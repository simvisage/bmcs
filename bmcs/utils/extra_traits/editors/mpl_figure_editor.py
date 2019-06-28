

from traits.etsconfig.api import ETSConfig

if ETSConfig.toolkit == 'wx':
    from .mpl_figure_editor_wx import MPLFigureEditor
if ETSConfig.toolkit == 'qt4':
    from .mpl_figure_editor_qt import MPLFigureEditor
else:
    raise ImportError("MPLEditor for %s toolkit not availabe" % \
        ETSConfig.toolkit)
