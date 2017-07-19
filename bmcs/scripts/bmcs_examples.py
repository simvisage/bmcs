'''
Created on Jul 17, 2017

@author: rch
'''

from reporter import Reporter
import part1_debonding.e21_bond_slip_damage
import part1_debonding.e22_bond_slip_plasticity
import part1_debonding.e23_bond_slip_damage_plasticity
import part1_debonding.e31_pullout_frictional
import part1_debonding.e32_pullout_multilinear
import part1_debonding.e33_pullout_frp_damage
import part1_debonding.e43_po_hardening_length_dependence
import part1_debonding.e44_po_softening_length_dependence

if __name__ == '__main__':

    r = Reporter()
    r.studies = [
        part1_debonding.e21_bond_slip_damage.construct_bond_slip_study(),
        part1_debonding.e22_bond_slip_plasticity.construct_bond_slip_study(),
        part1_debonding.e23_bond_slip_damage_plasticity.construct_bond_slip_study(),
        part1_debonding.e31_pullout_frictional.construct_pullout_study(),
        part1_debonding.e32_pullout_multilinear.construct_pullout_study(),
        part1_debonding.e33_pullout_frp_damage.construct_pullout_study(),
        part1_debonding.e43_po_hardening_length_dependence.PSLengthDependence(),
        part1_debonding.e44_po_softening_length_dependence.PSLengthDependence()
    ]
    r.write()
    r.show_tex()
    r.run_pdflatex()
    r.show_pdf()
