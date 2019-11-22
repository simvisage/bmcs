
from traits.api import \
    HasStrictTraits, \
    List, \
    on_trait_change, Property, cached_property, \
    Event
import numpy as np


class FEDomain(HasStrictTraits):
    '''Test the state dependencies within the hierarchical domain representation.
    '''

    changed_structure = Event

    subdomains = List(domain_changed=True)

    @on_trait_change('changed_structure')
    def _validate_subdomains(self):
        for domain in self.subdomains:
            domain.validate()

    serialized_subdomains = Property(depends_on='subdomains, subdomains_items')

    def _get_serialized_subdomains(self):
        '''Link the new subdomain at the end of the series.
        '''
        s = np.array(self.subdomains)
        for s1, s2 in zip(s[:-1], s[1:]):
            s1.xdomain.mesh.next_domain = s2
            s2.xdomain.mesh.previous_domain = s1
        return self.subdomains

    nonempty_subdomains = Property(depends_on='changed_structure')

    @cached_property
    def _get_nonempty_subdomains(self):
        d_list = []
        for d in self.serialized_subdomains:
            if d.xdomain.mesh.n_active_elems > 0:
                d_list.append(d)
        return d_list

    n_dofs = Property

    def _get_n_dofs(self):
        '''Return the total number of dofs in the domain.
        Use the last subdomain's: dof_offset + n_dofs 
        '''
        last_d = self.serialized_subdomains[-1]
        return last_d.xdomain.mesh.dof_offset + last_d.xdomain.mesh.n_dofs

    dof_offset_arr = Property

    def _get_dof_offset_arr(self):
        '''
        Return array of the dof offsets 
        from serialized subdomains
        '''
        a = np.array([domain.xdomain.mesh.dof_offset
                      for domain in self.serialized_subdomains])
        return a

    U_var_shape = Property

    def _get_U_var_shape(self):
        return (self.n_dofs,)

    def __iter__(self):
        return iter(self.subdomains)
