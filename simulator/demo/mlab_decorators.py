
from mayavi import mlab
import numpy as np


def decorate_figure(f, viz, distance=400, focal_point=(150, 40, 0)):
    mlab.view(0, 0, distance, np.array(focal_point), figure=f)
    mlab.orientation_axes(viz.src, figure=f)
    axes = mlab.axes(viz.src, figure=f)
    axes.label_text_property.trait_set(
        font_family='times', italic=False
    )
    axes.title_text_property.font_family = 'times'
    axes.axes.trait_set(
        x_label='x', y_label='y', z_label='z'
    )
