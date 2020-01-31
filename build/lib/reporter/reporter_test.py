'''

   1 - recursivity
   2 - figures
   3 - sort
   
   package dependency - window depends on reporter
   parametric studies?
'''
from reporter import Reporter
from traits.api import \
    Float,\
    Instance, Int

from report_item import \
    RInputRecord, RInputSection


class CrossSection(RInputRecord):
    name = 'Cross section'
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
    name = 'Geometry'
    length = Float(10.0, GEO=True, unit='mm', symbol='L')
    width = Float(0.4, GEO=True, unit='mm', symbol='b')


class ModelPart(RInputSection):
    name = 'part1'
    n_e = Int(20, GEO=True)
    cs = Instance(CrossSection, (), report=True)
    geo = Instance(Geometry, (), report=True)


if __name__ == '__main__':

    r = Reporter()
    r.report_items = [
        ModelPart(name='Part 1'),
    ]

    r.write()
    r.show_tex()
    r.run_pdflatex()
    r.show_pdf()
