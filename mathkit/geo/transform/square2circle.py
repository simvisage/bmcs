#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Jan 20, 2011 by: rch

from numpy import \
    array, tensordot, dot, zeros, c_, ix_, mgrid, arange, \
    where, sum, sin, cos, vstack, hstack, argmax, newaxis, size, \
    shape, sqrt, frompyfunc, ones_like, zeros_like, ones, any, all, \
    sort, argsort, concatenate, add
from numpy import where, zeros_like, fabs, sign, ma
from traits.api import \
    HasTraits, Float, Array, Property, cached_property, Instance, Enum, \
    Dict, Bool, Int, Callable


class GeoSquare2Circle(HasTraits):
    '''Geometry definition.
    '''
    circle_center = Array('float_', value=[0.0, 0.0])
    circle_radius = Float(0.38)
    square_edge = Float(2.0)

    pre_transform = Callable

    post_transform = Callable

    def __call__(self, points):
        if self.pre_transform:
            points = self.pre_transform(points)

        r, s = self._get_axes_data(points)
        r_t, s_t = self._transform(r, s)
        self._set_axes_data(points, r_t, s_t)

        if self.post_transform:
            points = self.post_transform(points)

        return points

    def _get_axes_data(self, points):
        return points[:, 0], points[:, 1]

    def _set_axes_data(self, points, r, s):
        points[:, 0] = r
        points[:, 1] = s

    def _transform(self, r, s):
        '''Return the global coordinates of the supplied local points.
        '''

        R = self.circle_radius
        D = self.square_edge / 2.0

        cx, cy = self.circle_center

        # number of local grid points for each coordinate direction
        # values must range between 0 and 1
        Xi, Yi = r - cx, s - cy

        # delimit the area within the square edge
        mask = where((fabs(Xi) <= D) & (fabs(Yi) <= D))

        xi, yi = Xi[ mask ], Yi[ mask ]
        X, Y = zeros_like(xi), zeros_like(yi)

        idx_center = where(xi ** 2 + yi ** 2 == 0)

        # size of total structure
        #
        idx_Rx = where(fabs(xi) > fabs(yi))
        idx_Ry = where(fabs(yi) >= fabs(xi))

        Px, Py = xi[ idx_Rx ], yi[ idx_Rx ]
        SPxy2 = sqrt(Px ** 2 + Py ** 2)

        X[idx_Rx] = sign(Px) * Px ** 2 / SPxy2
        Y[idx_Rx] = sign(Py) * fabs(Px) * fabs(Py) / SPxy2

        Px, Py = yi[ idx_Ry ], xi[ idx_Ry ]
        SPxy2 = sqrt(Px ** 2 + Py ** 2)

        Y[idx_Ry] = sign(Px) * Px ** 2 / SPxy2
        X[idx_Ry] = sign(Py) * fabs(Px) * fabs(Py) / SPxy2

        X[ idx_center ] = 0.0
        Y[ idx_center ] = 0.0

        delta_x, delta_y = X - xi, Y - yi

        # perform the transition mapping from the circle to the square

        idx_corner = where((fabs(xi) >= R) & (fabs(yi) >= R))
        idx_right = where((fabs(yi) < R) & (fabs(xi) >= R))
        idx_top = where((fabs(xi) < R) & (fabs(yi) >= R))

        xii, yii = xi[ idx_corner], yi[idx_corner ]
        print(R)
        print(D)
        a = 1.0 / (R ** 2 - 2 * D * R + D ** 2)
        b = -D / (R ** 2 - 2 * D * R + D ** 2)
        c = -D / (R ** 2 - 2 * D * R + D ** 2)
        d = D ** 2 / (R ** 2 - 2 * D * R + D ** 2)
        weight = (a * fabs(xii) * fabs(yii) + b * fabs(xii) + c * fabs(yii) + d)
        dx, dy = delta_x[ idx_corner ], delta_y[idx_corner]
        X[ idx_corner ], Y[ idx_corner ] = xii + weight * dx, yii + weight * dy

        xii, yii = xi[ idx_right], yi[idx_right ]
        weight = 1.0 / (R - D) * (fabs(xii) - D)
        dx, dy = delta_x[ idx_right ], delta_y[idx_right]
        X[ idx_right ], Y[ idx_right ] = xii + weight * dx, yii + weight * dy

        xii, yii = xi[ idx_top], yi[idx_top ]
        weight = 1.0 / (R - D) * (fabs(yii) - D)
        dx, dy = delta_x[ idx_top ], delta_y[idx_top]
        X[ idx_top ], Y[ idx_top ] = xii + weight * dx, yii + weight * dy

        Xi[mask] = X
        Yi[mask] = Y

        return Xi + cx, Yi + cy


if __name__ == '__main__':

    s2c = GeoSquare2Circle(circle_center=[0.2, 0.2], circle_radius=0.4, square_edge=2.0)
    print('result', s2c(array([[ 0.3, 0.5, 0 ]], dtype=float)))
