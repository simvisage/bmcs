
# Define the derivatives of the reference subspace
# with respect to the embedding space

from mathkit import EPS
import mayavi.mlab as mlab
import numpy as np
import sympy as sp
import traits.api as tr

eta_0, eta_1, eta_2 = sp.symbols('eta_0, eta_1, eta_2')


class Geo(tr.HasStrictTraits):

    eta = tr.List([eta_0, eta_1, eta_2])

    R_r = tr.List([sp.sin(eta_0) * eta_0, sp.cos(eta_0 + 1)**2, eta_1])

    def F_r(self, *eta_):
        r = np.array([
            re.subs({e: et for e, et in zip(self.eta, eta_)})
            for re in self.R_r
        ], dtype=np.float_)
        return np.einsum('ij,jk->ki', r[:, np.newaxis], np.identity(3))

    M_ra = tr.Property(depends_on='R_r, R_r_items')

    @tr.cached_property
    def _get_M_ra(self):
        return [[sp.diff(r, e) for r in self.R_r] for e in self.eta]

    def F_m(self, *eta_):
        return np.array(
            [
                [
                    m.subs({e: et for e, et in zip(self.eta, eta_)})
                    for m in m_a
                ]
                for m_a in self.M_ra
            ],
            dtype=np.float_
        )

    def orthogonal_base(self, *eta_):
        m_01 = self.F_m(*eta_)
        m_2 = np.einsum('...i,...j,ijk->...k', m_01[0, :], m_01[1, :], EPS)
        m_1a = np.einsum('...i,...j,ijk->...k', m_2, m_01[0, :], EPS)
        m = np.array([m_01[0, :], m_1a, m_2])
        m1 = m / np.sqrt(np.einsum('ij,ij->i', m, m))[:, np.newaxis]
        return m1

    def plot_basis(self, *eta):
        r = self.F_r(*eta)
        m = self.orthogonal_base(*eta)
        rm = r + m
        mp = np.vstack([r[np.newaxis, ...],
                        rm[np.newaxis, ...]])
        mlab.plot3d(mp[:, 0, 0], mp[:, 0, 1], mp[:, 0, 2],
                    color=black, tube_radius=0.01)
        mlab.plot3d(mp[:, 1, 0], mp[:, 1, 1], mp[:, 1, 2],
                    color=black, tube_radius=0.01)
        mlab.plot3d(mp[:, 2, 0], mp[:, 2, 1], mp[:, 2, 2],
                    color=black, tube_radius=0.01)
        mlab.text3d(rm[0, 0], rm[0, 1], rm[0, 2],
                    'X', color=black, scale=0.1)
        mlab.text3d(rm[1, 0], rm[1, 1], rm[1, 2],
                    'Y', color=black, scale=0.1)
        mlab.text3d(rm[2, 0], rm[2, 1], rm[2, 2],
                    'Z', color=black, scale=0.1)


g = Geo()

g.R_r = [eta_0, eta_1 * eta_2, eta_0 * eta_1]
g.R_r = [sp.sin(eta_0) * eta_0, sp.cos(eta_0 + eta_1 + 1)**2, eta_1 + eta_2]
#g.R_r = [sp.sin(eta_0), sp.cos(eta_0), eta_1]

black = (0, 0, 0)
white = (1, 1, 1)

ee0, ee1, ee2 = np.mgrid[0:2:10j, 0:1:5j, 0:1:5j]

mlab.figure(bgcolor=white)
for e0, e1, e2 in zip(ee0.flatten(), ee1.flatten(), ee2.flatten()):
    g.plot_basis(e0, e1, e2)

mlab.show()
