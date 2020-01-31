'''
Created on Jul 17, 2017

@author: rch
'''

from reporter import Reporter
from . import part1_debonding.e21_bond_slip_damage
from . import part1_debonding.e22_bond_slip_plasticity
from . import part1_debonding.e23_bond_slip_damage_plasticity
from . import part1_debonding.e31_pullout_frictional
from . import part1_debonding.e32_pullout_multilinear
from . import part1_debonding.e33_pullout_frp_damage
from . import part1_debonding.e43_po_hardening_length_dependence
from . import part1_debonding.e44_po_softening_length_dependence
from . import part2_cracking.e51_localization_zone

if __name__ == '__main__':

    r = Reporter()
    r.studies = [
        part1_debonding.e21_bond_slip_damage.BondSlipDamageStudy(),
        part1_debonding.e22_bond_slip_plasticity.BondSlipPlasticityStudy(),
        part1_debonding.e23_bond_slip_damage_plasticity.BondSlipDamagePlasticityStudy(),
        part1_debonding.e31_pullout_frictional.PullOutConstantBondStudy(),
        part1_debonding.e32_pullout_multilinear.PullOutMultilinearBondStudy(),
        part1_debonding.e33_pullout_frp_damage.PullOutFRPDamageBondStudy(),
        #         part1_debonding.e43_po_hardening_length_dependence.PSLengthDependenceStudy(),
        #         part1_debonding.e44_po_softening_length_dependence.PSLengthDependenceStudy(),
        #        part2_cracking.e51_localization_zone.LocalizationZoneStudy()
    ]
    r.write()
    r.show_tex()
    r.run_pdflatex()
    r.show_pdf()
