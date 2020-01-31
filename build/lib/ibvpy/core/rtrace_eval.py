
from traits.api import Array, Bool, Callable, Enum, Float, HasTraits, \
    Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
    Dict, Property, cached_property, WeakRef, Delegate
from traitsui.api import Item, View, HGroup, ListEditor, VGroup, VSplit, Group, HSplit

#from etsproxy.pyface.tvtk.actor_editor import ActorEditor
from .i_tstepper_eval import ITStepperEval


class RTraceEval(HasTraits):
    name = Str('unnamed')
    ts = WeakRef(ITStepperEval)

    u_mapping = Callable
    eval = Callable

    def __call__(self, sctx, u, *args, **kw):

        # When crossing the levels - start a mapping
        # This method might have side effects for the context
        # - mapping of global to local values
        #
        args_mapped = []
        kw_mapped = {}

        if self.u_mapping:

            u = self.u_mapping(sctx, u)

            # map everything that has been sent together with u
            # this might be the time derivatives of u or its
            # spatial integrals.
            #
            args_mapped = [self.u_mapping(sctx, u_value)
                           for u_value in args]

            kw_mapped = {}
            for u_name, u_value in list(kw.items()):
                kw_mapped[u_name] = self.u_mapping(sctx, u_value)

        # Invoke the tracer evaluation.
        #
        try:
            val = self.eval(sctx, u, *args_mapped, **kw_mapped)
        except TypeError as e:
            raise TypeError('tracer name %s: %s %s' % (
                self.name, e, self.eval))

        return val
