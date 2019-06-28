# Demonstrate the derivation of the orthonormal basis
# of a general discretization

from ibvpy.fets import FETS2D4Q
from ibvpy.mesh.fe_grid import FEGrid
from mathkit import EPS
import mayavi.mlab as m
import numpy as np


def geo_trans(points):
    '''Transform a grid geometry to alayered, 
    sinusoidal discretization.
    '''
    x, y = points.T
    return np.c_[x, np.sin(x) + y]


fets = FETS2D4Q()
mesh = FEGrid(coord_max=(6, 4),
              shape=(20, 2),
              fets_eval=fets,
              geo_transform=geo_trans
              )

To3D = fets.vtk_expand_operator

x_Ia = mesh.X_Id
X_Ia = np.einsum('...i,...ij->...j', x_Ia, To3D)
I_Ei = mesh.I_Ei
x_Eia = X_Ia[I_Ei, :]
J_Emar = np.einsum(
    'imr,...ia->...mar', fets.dN_imr, x_Eia
)
m_0_Ema = J_Emar[..., 0]
m_2_Ema = np.einsum(
    '...i,...j,ijk->...k',
    m_0_Ema, J_Emar[..., 1], EPS
)
m_1_Ema = np.einsum(
    '...i,...j,ijk->...k',
    m_2_Ema, m_0_Ema, EPS)
M_rEma = np.array([m_0_Ema, m_1_Ema, m_2_Ema])
M_Emra = np.einsum('rEma->Emra', M_rEma)
norm_M_Emra = np.sqrt(
    np.einsum('...ij,...ij->...i', M_Emra, M_Emra)
)[..., np.newaxis]

T_Emra = M_Emra / norm_M_Emra
x_Ema = np.einsum(
    'im,...ia->...ma', fets.N_im, x_Eia
)

m.figure(bgcolor=(1, 1, 1))
colors = [(.8, .8, .8), (.8, .8, .8), (.8, .8, .8)]
for r, c in zip((0, 1, 2), colors):
    vec = m.quiver3d(x_Ema[..., 0], x_Ema[..., 1], x_Ema[..., 2],
                     T_Emra[..., r, 0], T_Emra[..., r, 1], T_Emra[..., r, 2],
                     color=c)
    vec.glyph.glyph.scaling = True
    vec.glyph.glyph.clamping = False
    vec.glyph.glyph.scale_mode = 'scale_by_vector'

# consider a constant displacement field and transform it into
# the coordinates - visualize
u_Ema = .5 * np.ones_like(x_Ema)
u_Ema[..., (1, 2)] = 0

u_Emr = np.einsum(
    '...ra,...a->...r',
    T_Emra, u_Ema
)

uT_Emra = np.einsum(
    '...r,...ra->...ra',
    u_Emr, T_Emra
)
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
for r, c in zip((0, 1, 2), colors):
    vec = m.quiver3d(
        x_Ema[..., 0], x_Ema[..., 1], x_Ema[..., 2],
        uT_Emra[..., r, 0], uT_Emra[..., r, 1], uT_Emra[..., r, 2],
        color=c, scale_mode='vector')
    vec.glyph.glyph.scaling = True
    vec.glyph.glyph.clamping = False
    vec.glyph.glyph.scale_mode = 'scale_by_vector'
m.show()
