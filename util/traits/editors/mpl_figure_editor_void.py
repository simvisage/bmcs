
from traits.api import HasTraits


class _MPLFigureEditor(HasTraits):

    def init(self, parent):
        pass

    def update_editor(self):
        pass


class MPLFigureEditor(HasTraits):

    klass = _MPLFigureEditor
