'''

   1 - recursivity
   2 - figures
   3 - sort
   
   package dependency - window depends on reporter
   parametric studies?
'''
from traits.api import \
    Float,\
    Instance

from report_item import \
    RInputRecord, RInputSubDomain
from reporter import Reporter


class CrossSection(RInputRecord):

    A_m = Float(0.4, CS=True, unit='mm',
                label='Area of the matrix',
                math_symbol='A_\mathrm{m}')

    A_f = Float(0.4, CS=True, unit='mm',
                label='Area of the reinforcement',
                math_symbol='A_\mathrm{f}')

    P_b = Float(0.2, CS=True, unit='mm',
                label='Contact perimeter',
                math_symbol='P_\mathrm{b}')


class Geometry(RInputRecord):

    length = Float(10.0, GEO=True, unit='mm', symbol='L')
    width = Float(0.4, GEO=True, unit='mm', symbol='b')


class ModelPart(RInputSubDomain):
    name = 'part1'
    cs = Instance(CrossSection, (), CS=True)
    geo = Instance(Geometry, (), GEO=True)


if __name__ == '__main__':

    r = Reporter()
    r.report_items = [
        ModelPart(name='Part 1'),
    ]

    r.write()
    r.show_tex()
    r.run_pdflatex()
    r.show_pdf()
