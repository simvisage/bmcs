'''
Created on Jan 31, 2018

@author: rch
'''

from mayavi import mlab

import numpy as np
x, y = np.mgrid[0:3:1, 0:3:1]
s = mlab.surf(x, y, np.asarray(x * 0.1, 'd'))


@mlab.show
@mlab.animate(delay=250)
def anim():
    """Animate the b1 box."""
    for i in range(10):
        s.mlab_source.scalars = np.asarray(x * 0.1 * (i + 1), 'd')
        yield


# Run the animation.
anim()
