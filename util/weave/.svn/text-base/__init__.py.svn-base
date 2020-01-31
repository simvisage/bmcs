#
# weave - C/C++ integration
#

from inline_tools import inline
from numpy.testing import Tester
from scipy.weave import converters
from scipy.weave.ext_tools import ext_module, ext_function
from scipy.weave.info import __doc__
from scipy.weave.weave_version import weave_version as __version__
import scipy.weave.ext_tools as ext_tools

try:
    from scipy.weave.blitz_tools import blitz
except ImportError:
    pass # scipy (core) wasn't available

test = Tester().test
