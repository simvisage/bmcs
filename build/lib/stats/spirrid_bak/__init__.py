import warnings
warnings.warn("This SPIRRID module is deprecated.", category=UserWarning, stacklevel=2)
from .i_rf import IRF
from .rf import RF
from .rv import RV
from .spirrid import SPIRRID, make_ogrid
from mathkit.numpy.numpy_func import Heaviside
