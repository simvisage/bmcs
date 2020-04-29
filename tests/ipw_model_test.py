'''

   1 - recursivity
   2 - figures
   3 - sort
   
   package dependency - window depends on reporter
   parametric studies?
'''
from reporter.report_item import \
    RInputRecord, ROutputSection
from traits.api import \
    Float,\
    Instance, Int


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
    cs = Instance(CrossSection, (), ipw=True)
    geo = Instance(Geometry, (), ipw=True)


if __name__ == '__main__':

    INPUTTAGS = ['MAT', 'GEO', 'CS', 'ALG', 'BC', 'MESH']

    itags = {tag: True for tag in INPUTTAGS}

    mp = ModelPart()

    def get_ipw_model_components(mp):
        yield mp
        for tr in mp.trait_get(ipw=True).values():
            yield tr
            get_ipw_model_components(tr)

    ipw_components = get_ipw_model_components(mp)

    def get_itag_traits(mp_traits):
        for n_tr in mp_traits:
            for itag in list(itags.keys()):
                for name, tr in n_tr.traits(**{itag: True}).items():
                    if tr.label == None:
                        tr.label = name
                    yield tr

    ipw_traits = get_itag_traits(ipw_components)

    for ipw_trait in ipw_traits:
        print(ipw_trait.label)
