from traits.api import \
    HasTraits, provides, \
    Instance, Int, Str, \
    on_trait_change, Property, \
    self, TraitError
from traitsui.api import View, Item, Group, Include

from .fe_domain import FEDomain
from .i_fe_subdomain import IFESubDomain


_subdomain_counter = 0


@provides(IFESubDomain)
class FESubDomain(HasTraits):

    # If the constructor specifies the name, then use it
    # otherwise generate the name based on the domain enumeration.
    _name = Str('_no_name_')
    _tree_label = Str('subdomain')

    tstepper = Property

    def _get_tstepper(self):
        return self.domain.dots.tstepper

    # specification of the container domain
    # managing all the sub domains
    _domain = Instance(FEDomain)
    domain = Property

    def _set_domain(self, value):
        'reset the domain of this domain'
        if self._domain:
            # unregister in the old domain
            raise NotImplementedError(
                'FESubDomain cannot be relinked to another FEDomain')

        self._domain = value
        # register in the domain as a sub domain
        self._domain.subdomains.append(self)
        self._domain._append_in_series(self)

    def _get_domain(self):
        return self._domain

    def validate(self):
        pass

    name = Property(Str)

    def _get_name(self):
        global _subdomain_counter
        if self._name == '_no_name_':
            self._name = self._tree_label + ' ' + str(_subdomain_counter)
            _subdomain_counter += 1
        return self._name

    def _set_name(self, value):
        self._name = value

    # local dof enumeration
    n_dofs = Int(4, domain_changed=True)

    # dof offset within the global enumeration
    dof_offset = Property(Int, depends_on='previous_domain.dof_offset')
    # cached_property

    def _get_dof_offset(self):
        if self.previous_domain:
            return self.previous_domain.dof_offset + self.previous_domain.n_dofs
        else:
            return 0

    def __repr__(self):
        if self.previous_domain:
            return self.name + ' <- ' + self.previous_domain.name
        else:
            return self.name

    # get the slice for DOFs within global vectors
    sub_slice = Property

    def _get_sub_slice(self):
        return slice(self.dof_offset, self.dof_offset + self.n_dofs)

    # dependency link for sequential enumeration
    previous_domain = Instance(IFESubDomain, domain_changed=True)

    @on_trait_change('previous_domain')
    def _validate_previous_domain(self):
        if self.previous_domain == self:
            raise TraitError('cyclic reference for ' + self.name)

    # dependency link for sequential enumeration
    next_domain = Instance(IFESubDomain, domain_changed=True)

    @on_trait_change('next_domain')
    def _validate_next_domain(self):
        if self.next_domain == self:
            raise TraitError('cyclic reference for ' + self.name)

    subdomain_group = Group(Item('n_dofs'),
                            Item('dof_offset'),
                            Item('previous_domain'),
                            Item('next_domain'),
                            )

    traits_view = View(Include('subdomain_group'),
                       resizable=True,
                       scrollable=True
                       )
