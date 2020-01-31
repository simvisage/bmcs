'''

   1 - recursivity
   2 - figures
   3 - sort
   
   package dependency - window depends on reporter
   parametric studies?
'''
from .reporter import Reporter, ReportStudy
from traits.api import \
    Float,\
    Instance, Int

from .report_item import \
    RInputRecord, ROutputSection


class CrossSection(RInputRecord):
    name = 'Cross section'
    A_m = Float(0.4, CS=True, unit='mm',
                label='Area of the matrix',
                symbol='A_\mathrm{m}')

    A_f = Float(0.4, CS=True, unit='mm',
                label='Area of the reinforcement',
                symbol='A_\mathrm{f}')

    P_b = Float(0.2, CS=True, unit='mm',
                label='Contact perimeter',
                symbol='P_\mathrm{b}')


class Geometry(RInputRecord):
    name = 'Geometry'
    length = Float(10.0, GEO=True, unit='mm', symbol='L')
    width = Float(0.4, GEO=True, unit='mm', symbol='b')


class ModelPart(RInputRecord):
    name = 'part1'
    title = 'Model part 1'
    desc = 'Some description'
    n_e = Int(20, symbol='n_\mathrm{e}', GEO=True)
    cs = Instance(CrossSection, (), report=True)
    geo = Instance(Geometry, (), report=True)


if __name__ == '__main__':

    r = Reporter()
    r.studies = [
        ReportStudy(
            input=ModelPart(name='part_1')
        )
    ]
    r.write()
    r.show_tex()
    r.run_pdflatex()
    r.show_pdf()
