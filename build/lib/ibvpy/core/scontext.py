
class SContext:  # (HasTraits):

    '''
    Spatial context represents a complex reference within the
    spatial object.

    In particular, spatial context of a particular material point is
    represented as tuple containing tuple of references to [domain,
    element, layer, integration cell, material point]

    The context is filled when stepping over the discretization
    levels. It is included in all parameters of the time-step-evals
    and resp-trace-evals.
    '''
