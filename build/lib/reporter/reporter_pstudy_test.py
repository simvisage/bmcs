'''

   1 - recursivity
   2 - figures
   3 - sort
   
   package dependency - window depends on reporter
   parametric studies?
'''
from .reporter import Reporter
from traits.api import \
    Float,\
    Instance, Int

from .report_item import \
    RInputRecord, RInputSection, ROutputSection


class CrossSection(RInputSection):
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


class Geometry(RInputSection):
    name = 'Geometry'
    length = Float(10.0, GEO=True, unit='mm', symbol='$L$')
    width = Float(0.4, GEO=True, unit='mm', symbol='$b$')


class ModelInput(RInputSection):
    name = 'part1'
    title = 'Model part 1'
    desc = 'Testing p-study'
    n_e = Int(20, GEO=True)
    cross_section = Instance(CrossSection, (), report=True)
    geo = Instance(Geometry, (), report=True)


class ModelOutput(ROutputSection):
    model_input = Instance(ModelInput)
    name = 'part1'
    title = 'Model part 1'
    desc = 'Testing p-study'

    def get_records(self):
        return []


def generate_report():
    mpi = ModelInput(name='Part 1')

    mpo = ModelOutput()
    r = Reporter()
    r.report_items = [mpi, mpo
                      ]
    rinput_tree = r.yield_rinput_traits()

    for path, trait in list(rinput_tree.items()):
        print(path, trait)
        trait.value_range = [2, 3, 4, 5]

    r.write()
    r.show_tex()
    r.run_pdflatex()
    r.show_pdf()


if __name__ == '__main__':
    rinput_tree = generate_report()
