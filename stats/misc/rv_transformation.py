'''
Created on 20.9.2011

numerical transformation of a random variable defined as x_values and
corresponding pdf_x values of the density function;
the object needs a y_values domain to evaluate the transformed X --> Y pdf at
and a transformation method transf_eqn mapping X to Y (y = x**2 in the example) 

@author: Q
'''

from etsproxy.traits.api import \
    HasTraits, Array, Property, cached_property, \
    Callable

from mathkit.mfn import MFnLineArray
import scipy as sp
import numpy as np

class RVTransformation(HasTraits):

    x_values = Array
    pdf_x = Array
    # TODO: the possibility to work with CDFs
    cdf_x = Array

    PDFx = Property(depends_on = 'x_values, pdf_x')
    @cached_property
    def _get_PDFx(self):
        pdfx_line = MFnLineArray(xdata = self.x_values, ydata = self.pdf_x)
        return pdfx_line

    transf_eqn = Callable
    def _transf_eqn_default(self):
        # define simple linear transfomation function
        return lambda x: x

    y_values = Property(Array, depends_on = 'x_values, transf_eqn')
    @cached_property
    def _get_y_values(self):
        y = self.transf_eqn(self.x_values)
        y = np.linspace(y.min(), y.max(), len(self.x_values))
        return y

    transf_line = Property(depends_on = 'x_values, pdf_x, transf_eqn')
    @cached_property
    def _get_transf_line(self):
        x = np.linspace(self.x_values[0], self.x_values[-1], 500)
        y = self.transf_eqn(x)
        return MFnLineArray(xdata = x, ydata = y, extrapolate = 'diff')

    def transf_diff(self, x):
        return self.transf_line.get_diffs(x, k = 1, der = 1)

    def transf_inv_scalar(self, y):
        x = np.linspace(self.x_values[0], self.x_values[-1], 500)
        s = sp.interpolate.InterpolatedUnivariateSpline(x, self.transf_eqn(x) - y)
        roots = s.roots()
        return roots

    def transf_inv_vect(self, y):
        res = []
        y_val = []
        for yi in y:
            value = self.transf_inv_scalar(yi)
            if len(value) != 0:
                res.append(value)
                y_val.append(yi)
        return res, y_val

    PDFy = Property(depends_on = 'x_values, pdf_x, transf_eqn')
    @cached_property
    def _get_PDFy(self):
        x, y = self.transf_inv_vect(self.y_values)
        xdata = []
        PDFy = []
        for i, xi in enumerate(x):
            if np.abs(self.transf_diff(np.array(xi))).all() > 1e-15:
                PDFy_arr = self.PDFx.get_values(np.array(xi), k = 1) / np.abs(self.transf_diff(np.array(xi)))
                PDFy.append(np.sum(PDFy_arr))
                xdata.append(y[i])
        return MFnLineArray(xdata = xdata, ydata = PDFy)

if __name__ == '__main__':

    from math import e
    from matplotlib import pyplot as plt
    from scipy.integrate import simps

    #===========================
    # RV TRANSFORMATION EXAMPLE
    #===========================

    # definition of the initial PDF(x)
    def H(x):
        return np.sign(np.sign(x) + 1)

    def pdfx (x):
        a = 1 / 2.
        b = 0.5 * e ** (-x)
        return a * H(x + 1) * H(-x) + b * H(x)

    x = np.linspace(-7, 7, 1000)

    # construction of the transformation object and definition of its attr
    trnsf = RVTransformation(x_values = x,
                              pdf_x = pdfx(x)
                              )

    # 1. definition of the transformation equation
    def tr1(x):
        return np.sin(x / 2.)
    trnsf.transf_eqn = tr1
    # plot
    plt.plot(trnsf.y_values, trnsf.PDFy.get_values(trnsf.y_values), lw = 2, label = 'transf $y = \sin(x/2)$')

    # 2. definition of the transformation equation
    def tr2(x):
        return 2 * x
    trnsf.transf_eqn = tr2
    plt.plot(trnsf.y_values, trnsf.PDFy.get_values(trnsf.y_values), lw = 2, label = 'transf $y = 2x$')

    # 3. definition of the transformation equation
    def tr3(x):
        return x ** 2
    trnsf.transf_eqn = tr3
    plt.plot(trnsf.y_values, trnsf.PDFy.get_values(trnsf.y_values), lw = 2, label = 'transf $y = x^2$')

    plt.plot(x, trnsf.PDFx.get_values(x), lw = 2, color = 'black', label = 'initial PDF')
    plt.legend(loc = 'best')
    plt.xlabel('x')
    plt.ylabel('PDF(x)')
    plt.ylim(0, 2)
    plt.xlim(-2.3, 3)
    plt.show()

