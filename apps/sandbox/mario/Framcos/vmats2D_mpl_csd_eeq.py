'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 2D

(compressive plasticity (CP) + tensile damage (TD) 
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''

from traits.api import Constant, \
    Float, Dict, Property, cached_property

from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
import numpy as np
import traits.api as tr


class MATS2DMplCSDEEQ(MATS2DEval):

    # PARAMETERS C40

    # Tangential constitutive law parameters
    #---------------------------------------
    #     gamma_T = Float(30000.,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(1000.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.008,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(25,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #     e_T = Float(30.,
    #                 label="e",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(4,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(4.5,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.016,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(30.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     Ad2 = Float(50.0,
    #                 label="a",
    #                 desc="brittleness coefficient",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     eps_0 = Float(0.00001,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #     eps_02 = Float(1000.0013,
    #                    label="a",
    #                    desc="threshold strain",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(19000.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(19000.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(20.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     state_var_shapes = Property(Dict(), depends_on='n_mp')
    #     '''Dictionary of state variable entries with their array shapes.
    #     '''
    #
    #     #-------------------------------------------------------------------------
    #     # Cached elasticity tensors
    #     #-------------------------------------------------------------------------
    #
    #     E = tr.Float(35e+3,
    #                  label="E",
    #                  desc="Young's Modulus",
    #                  auto_set=False,
    #                  input=True)
    #
    #     nu = tr.Float(0.2,
    #                   label='nu',
    #                   desc="Poison ratio",
    #                   auto_set=False,
    #                   input=True)

    # gamma_T = Float(800000.,
    #                 label="Gamma",
    #                 desc=" Tangential Kinematic hardening modulus",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    # K_T = Float(50000.0,
    #             label="K",
    #             desc="Tangential Isotropic harening",
    #             enter_set=True,
    #             auto_set=False)
    #
    # S_T = Float(0.029,
    #             label="S",
    #             desc="Damage strength",
    #             enter_set=True,
    #             auto_set=False)
    #
    # r_T = Float(13.,
    #             label="r",
    #             desc="Damage cumulation parameter",
    #             enter_set=True,
    #             auto_set=False)
    #
    # c_T = Float(8,
    #             label="c",
    #             desc="Damage cumulation parameter",
    #             enter_set=True,
    #             auto_set=False)
    #
    # e_T = Float(11.,
    #             label="c",
    #             desc="Damage cumulation parameter",
    #             enter_set=True,
    #             auto_set=False)
    #
    # tau_pi_bar = Float(2.0,
    #                    label="Tau_bar",
    #                    desc="Reversibility limit",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    # a = Float(0.012,
    #           label="a",
    #           desc="Lateral pressure coefficient",
    #           enter_set=True,
    #           auto_set=False)
    #
    # # -------------------------------------------
    # # Normal_Tension constitutive law parameters (without cumulative normal strain)
    # # -------------------------------------------
    # Ad = Float(1000.0,
    #            label="a",
    #            desc="brittleness coefficient",
    #            enter_set=True,
    #            auto_set=False)
    #
    # eps_0 = Float(0.0001,
    #               label="a",
    #               desc="threshold strain",
    #               enter_set=True,
    #               auto_set=False)

    #     Ad = Float(100000.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     eps_0 = Float(1e-8,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)

    # -----------------------------------------------
    # Normal_Compression constitutive law parameters
    # -----------------------------------------------
    # K_N = Float(80000.,
    #             label="K_N",
    #             desc=" Normal isotropic harening",
    #             enter_set=True,
    #             auto_set=False)
    #
    # gamma_N = Float(100000.,
    #                 label="gamma_N",
    #                 desc="Normal kinematic hardening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    # sigma_0 = Float(80.,
    #                 label="sigma_0",
    #                 desc="Yielding stress",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    # state_var_shapes = Property(Dict(), depends_on='n_mp')
    # '''Dictionary of state variable entries with their array shapes.
    # '''
    #
    # # -------------------------------------------------------------------------
    # # Cached elasticity tensors
    # # -------------------------------------------------------------------------
    #
    # E = tr.Float(34e+3,
    #              label="E",
    #              desc="Young's Modulus",
    #              auto_set=False,
    #              input=True)
    #
    # nu = tr.Float(0.2,
    #               label='nu',
    #               desc="Poison ratio",
    #               auto_set=False,
    #               input=True)
    #     #    PARAMETERS FOR C80

    # gamma_T = Float(1000000.,
    #                 label="Gamma",
    #                 desc=" Tangential Kinematic hardening modulus",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    # K_T = Float(30000.0,
    #             label="K",
    #             desc="Tangential Isotropic harening",
    #             enter_set=True,
    #             auto_set=False)
    #
    # S_T = Float(0.011,
    #             label="S",
    #             desc="Damage strength",
    #             enter_set=True,
    #             auto_set=False)
    #
    # r_T = Float(16.,
    #             label="r",
    #             desc="Damage cumulation parameter",
    #             enter_set=True,
    #             auto_set=False)
    #
    # c_T = Float(6.,
    #             label="c",
    #             desc="Damage cumulation parameter",
    #             enter_set=True,
    #             auto_set=False)
    #
    # e_T = Float(14.,
    #             label="c",
    #             desc="Damage cumulation parameter",
    #             enter_set=True,
    #             auto_set=False)
    #
    # tau_pi_bar = Float(2.0,
    #                    label="Tau_bar",
    #                    desc="Reversibility limit",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    # a = Float(0.01,
    #           label="a",
    #           desc="Lateral pressure coefficient",
    #           enter_set=True,
    #           auto_set=False)
    #
    # #-------------------------------------------
    # # Normal_Tension constitutive law parameters (without cumulative normal strain)
    # #-------------------------------------------
    # Ad = Float(1000.0,
    #            label="a",
    #            desc="brittleness coefficient",
    #            enter_set=True,
    #            auto_set=False)
    #
    # Ad2 = Float(1000.0,
    #             label="a",
    #             desc="brittleness coefficient",
    #             enter_set=True,
    #             auto_set=False)
    #
    # eps_0 = Float(0.0001,
    #               label="a",
    #               desc="threshold strain",
    #               enter_set=True,
    #               auto_set=False)
    #
    # eps_02 = Float(400.003,
    #                label="a",
    #                desc="threshold strain",
    #                enter_set=True,
    #                auto_set=False)
    #
    # #-----------------------------------------------
    # # Normal_Compression constitutive law parameters
    # #-----------------------------------------------
    # K_N = Float(40000.,
    #             label="K_N",
    #             desc=" Normal isotropic harening",
    #             enter_set=True,
    #             auto_set=False)
    #
    # gamma_N = Float(30000.,
    #                 label="gamma_N",
    #                 desc="Normal kinematic hardening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    # sigma_0 = Float(15.,
    #                 label="sigma_0",
    #                 desc="Yielding stress",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    # state_var_shapes = Property(Dict(), depends_on='n_mp')
    # '''Dictionary of state variable entries with their array shapes.
    # '''

    #-------------------------------------------------------------------------
    # Cached elasticity tensors
    #-------------------------------------------------------------------------

    # E = tr.Float(42e+3,
    #              label="E",
    #              desc="Young's Modulus",
    #              auto_set=False,
    #              input=True)
    #
    # nu = tr.Float(0.2,
    #               label='nu',
    #               desc="Poison ratio",
    #               auto_set=False,
    #               input=True)

    #     #    PARAMETERS FOR C80 MA

    gamma_T = Float(1000000.,
                    label="Gamma",
                    desc=" Tangential Kinematic hardening modulus",
                    enter_set=True,
                    auto_set=False)

    K_T = Float(30000.0,
                label="K",
                desc="Tangential Isotropic harening",
                enter_set=True,
                auto_set=False)

    S_T = Float(0.01,
                label="S",
                desc="Damage strength",
                enter_set=True,
                auto_set=False)

    r_T = Float(14.,
                label="r",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    c_T = Float(6.,
                label="c",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    e_T = Float(14.,
                label="c",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    tau_pi_bar = Float(2.0,
                       label="Tau_bar",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    a = Float(0.01,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    #-------------------------------------------
    # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #-------------------------------------------
    Ad = Float(1000.0,
               label="a",
               desc="brittleness coefficient",
               enter_set=True,
               auto_set=False)

    Ad2 = Float(1000.0,
                label="a",
                desc="brittleness coefficient",
                enter_set=True,
                auto_set=False)

    eps_0 = Float(0.0001,
                  label="a",
                  desc="threshold strain",
                  enter_set=True,
                  auto_set=False)

    eps_02 = Float(400.003,
                   label="a",
                   desc="threshold strain",
                   enter_set=True,
                   auto_set=False)

    #-----------------------------------------------
    # Normal_Compression constitutive law parameters
    #-----------------------------------------------
    K_N = Float(30000.,
                label="K_N",
                desc=" Normal isotropic harening",
                enter_set=True,
                auto_set=False)

    gamma_N = Float(20000.,
                    label="gamma_N",
                    desc="Normal kinematic hardening",
                    enter_set=True,
                    auto_set=False)

    sigma_0 = Float(60.,
                    label="sigma_0",
                    desc="Yielding stress",
                    enter_set=True,
                    auto_set=False)

    state_var_shapes = Property(Dict(), depends_on='n_mp')
    '''Dictionary of state variable entries with their array shapes.
    '''

    #-------------------------------------------------------------------------
    # Cached elasticity tensors
    #-------------------------------------------------------------------------

    E = tr.Float(42e+3,
                 label="E",
                 desc="Young's Modulus",
                 auto_set=False,
                 input=True)

    nu = tr.Float(0.2,
                  label='nu',
                  desc="Poison ratio",
                  auto_set=False,
                  input=True)

    #    PARAMETERS FOR C120

    #     gamma_T = Float(2000000.,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(2200.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.01,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(7,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(4.,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     e_T = Float(6.,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(1.7,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.01,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(1000.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     Ad2 = Float(1000.0,
    #                 label="a",
    #                 desc="brittleness coefficient",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     eps_0 = Float(0.0001,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #
    #     eps_02 = Float(400.003,
    #                    label="a",
    #                    desc="threshold strain",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(8000.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(6000.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(140.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     state_var_shapes = Property(Dict(), depends_on='n_mp')
    #     '''Dictionary of state variable entries with their array shapes.
    #     '''
    #
    #     #-------------------------------------------------------------------------
    #     # Cached elasticity tensors
    #     #-------------------------------------------------------------------------
    #
    #     E = tr.Float(44e+3,
    #                  label="E",
    #                  desc="Young's Modulus",
    #                  auto_set=False,
    #                  input=True)
    #
    #     nu = tr.Float(0.2,
    #                   label='nu',
    #                   desc="Poison ratio",
    #                   auto_set=False,
    #                   input=True)

    #    PARAMETERS FOR C120 MA

    #     gamma_T = Float(2000000.,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(2200.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.015,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(17.5,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(8.,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     e_T = Float(10.,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(1.8,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.008,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(1000.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     Ad2 = Float(1000.0,
    #                 label="a",
    #                 desc="brittleness coefficient",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     eps_0 = Float(0.0001,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #
    #     eps_02 = Float(400.003,
    #                    label="a",
    #                    desc="threshold strain",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(35000.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(25000.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(90.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     state_var_shapes = Property(Dict(), depends_on='n_mp')
    #     '''Dictionary of state variable entries with their array shapes.
    #     '''
    #
    #     #-------------------------------------------------------------------------
    #     # Cached elasticity tensors
    #     #-------------------------------------------------------------------------
    #
    #     E = tr.Float(44e+3,
    #                  label="E",
    #                  desc="Young's Modulus",
    #                  auto_set=False,
    #                  input=True)
    #
    #     nu = tr.Float(0.2,
    #                   label='nu',
    #                   desc="Poison ratio",
    #                   auto_set=False,
    #                   input=True)

    # PARAMETERS PARAMETRIC STUDY FIGURE 5 FRAMCOS PAPER
    # TENSILE

    #     gamma_T = Float(80000.,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(10000.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.000000001,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(1.21,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #     e_T = Float(1.,
    #                 label="e",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(1.85,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(0.1,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.001,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(1500.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     Ad2 = Float(50.0,
    #                 label="a",
    #                 desc="brittleness coefficient",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     eps_0 = Float(0.00008,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #     eps_02 = Float(1000.0013,
    #                    label="a",
    #                    desc="threshold strain",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(4000.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(20000.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(180.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     state_var_shapes = Property(Dict(), depends_on='n_mp')
    #     '''Dictionary of state variable entries with their array shapes.
    #     '''
    #
    #     #-------------------------------------------------------------------------
    #     # Cached elasticity tensors
    #     #-------------------------------------------------------------------------
    #
    #     E = tr.Float(35e+3,
    #                  label="E",
    #                  desc="Young's Modulus",
    #                  auto_set=False,
    #                  input=True)
    #
    #     nu = tr.Float(0.2,
    #                   label='nu',
    #                   desc="Poison ratio",
    #                   auto_set=False,
    #                   input=True)

    # COMPRESSION

    #     gamma_T = Float(10000.,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(10000.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.000007,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(1.2,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #     e_T = Float(1.,
    #                 label="e",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(1.25,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(5.,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.001,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(1500.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     Ad2 = Float(15.0,
    #                 label="a",
    #                 desc="brittleness coefficient",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     eps_0 = Float(0.00008,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #     eps_02 = Float(8e4,
    #                    label="a",
    #                    desc="threshold strain",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(10000.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(10000.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(30.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     state_var_shapes = Property(Dict(), depends_on='n_mp')
    #     '''Dictionary of state variable entries with their array shapes.
    #     '''
    #
    #     #-------------------------------------------------------------------------
    #     # Cached elasticity tensors
    #     #-------------------------------------------------------------------------
    #
    #     E = tr.Float(35e+3,
    #                  label="E",
    #                  desc="Young's Modulus",
    #                  auto_set=False,
    #                  input=True)
    #
    #     nu = tr.Float(0.2,
    #                   label='nu',
    #                   desc="Poison ratio",
    #                   auto_set=False,
    #                   input=True)

    # BI-AXIAL ENVELOPE

    #     gamma_T = Float(10000.,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(10000.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.000007,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(1.2,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #     e_T = Float(1.,
    #                 label="e",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(1.8,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(5.,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.01,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(50000.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     Ad2 = Float(50.0,
    #                 label="a",
    #                 desc="brittleness coefficient",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     eps_0 = Float(0.00008,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #     eps_02 = Float(1000.0013,
    #                    label="a",
    #                    desc="threshold strain",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(15000.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(20000.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(30.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     state_var_shapes = Property(Dict(), depends_on='n_mp')
    #     '''Dictionary of state variable entries with their array shapes.
    #     '''
    #
    #     #-------------------------------------------------------------------------
    #     # Cached elasticity tensors
    #     #-------------------------------------------------------------------------
    #
    #     E = tr.Float(35e+3,
    #                  label="E",
    #                  desc="Young's Modulus",
    #                  auto_set=False,
    #                  input=True)
    #
    #     nu = tr.Float(0.2,
    #                   label='nu',
    #                   desc="Poison ratio",
    #                   auto_set=False,
    #                   input=True)

    #     gamma_T = Float(1000000.,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(30000.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.011,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(16.,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(8.,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     e_T = Float(4.,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(2.0,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.01,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(1000.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     Ad2 = Float(1000.0,
    #                 label="a",
    #                 desc="brittleness coefficient",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     eps_0 = Float(0.0001,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #
    #     eps_02 = Float(400.003,
    #                    label="a",
    #                    desc="threshold strain",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(40000.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(30000.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(15.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     state_var_shapes = Property(Dict(), depends_on='n_mp')
    #     '''Dictionary of state variable entries with their array shapes.
    #     '''
    #
    #     #-------------------------------------------------------------------------
    #     # Cached elasticity tensors
    #     #-------------------------------------------------------------------------
    #
    #     E = tr.Float(42e+3,
    #                  label="E",
    #                  desc="Young's Modulus",
    #                  auto_set=False,
    #                  input=True)
    #
    #     nu = tr.Float(0.2,
    #                   label='nu',
    #                   desc="Poison ratio",
    #                   auto_set=False,
    #                   input=True)

    # ALTERNATIVE C80

    # gamma_T = Float(20000.,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(1000.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.01,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(22.,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #     e_T = Float(4.,
    #                 label="e",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(8,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(5.0,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.005,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(800.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     Ad2 = Float(50.0,
    #                 label="a",
    #                 desc="brittleness coefficient",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     eps_0 = Float(0.00001,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #     eps_02 = Float(1000.0013,
    #                    label="a",
    #                    desc="threshold strain",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(20000.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(20000.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(40.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)

    #     gamma_T = Float(5000000.0,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(500000.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.0015,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(4,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #     e_T = Float(1.,
    #                 label="e",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(1.25,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(5.,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.01,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(1500.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     Ad2 = Float(15.0,
    #                 label="a",
    #                 desc="brittleness coefficient",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     eps_0 = Float(0.00008,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #     eps_02 = Float(8e4,
    #                    label="a",
    #                    desc="threshold strain",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(5000.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(2000.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(30.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)

    #     # C40 EXPERIMENTS

    # gamma_T = Float(100000.,
    #                 label="Gamma",
    #                 desc=" Tangential Kinematic hardening modulus",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    # K_T = Float(10000.0,
    #             label="K",
    #             desc="Tangential Isotropic harening",
    #             enter_set=True,
    #             auto_set=False)
    #
    # S_T = Float(0.005,
    #             label="S",
    #             desc="Damage strength",
    #             enter_set=True,
    #             auto_set=False)
    #
    # r_T = Float(9.,
    #             label="r",
    #             desc="Damage cumulation parameter",
    #             enter_set=True,
    #             auto_set=False)
    # e_T = Float(12.,
    #             label="e",
    #             desc="Damage cumulation parameter",
    #             enter_set=True,
    #             auto_set=False)
    #
    # c_T = Float(4.6,
    #             label="c",
    #             desc="Damage cumulation parameter",
    #             enter_set=True,
    #             auto_set=False)
    #
    # tau_pi_bar = Float(1.7,
    #                    label="Tau_bar",
    #                    desc="Reversibility limit",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    # a = Float(0.003,
    #           label="a",
    #           desc="Lateral pressure coefficient",
    #           enter_set=True,
    #           auto_set=False)
    #
    # #-------------------------------------------
    # # Normal_Tension constitutive law parameters (without cumulative normal strain)
    # #-------------------------------------------
    # Ad = Float(100.0,
    #            label="a",
    #            desc="brittleness coefficient",
    #            enter_set=True,
    #            auto_set=False)
    #
    # Ad2 = Float(15.0,
    #             label="a",
    #             desc="brittleness coefficient",
    #             enter_set=True,
    #             auto_set=False)
    #
    # eps_0 = Float(0.00008,
    #               label="a",
    #               desc="threshold strain",
    #               enter_set=True,
    #               auto_set=False)
    # eps_02 = Float(8e4,
    #                label="a",
    #                desc="threshold strain",
    #                enter_set=True,
    #                auto_set=False)
    #
    # #-----------------------------------------------
    # # Normal_Compression constitutive law parameters
    # #-----------------------------------------------
    # K_N = Float(10000.,
    #             label="K_N",
    #             desc=" Normal isotropic harening",
    #             enter_set=True,
    #             auto_set=False)
    #
    # gamma_N = Float(5000.,
    #                 label="gamma_N",
    #                 desc="Normal kinematic hardening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    # sigma_0 = Float(30.,
    #                 label="sigma_0",
    #                 desc="Yielding stress",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    # state_var_shapes = Property(Dict(), depends_on='n_mp')
    # '''Dictionary of state variable entries with their array shapes.
    # '''
    #
    # #-------------------------------------------------------------------------
    # # Cached elasticity tensors
    # #-------------------------------------------------------------------------
    #
    # E = tr.Float(35e+3,
    #              label="E",
    #              desc="Young's Modulus",
    #              auto_set=False,
    #              input=True)
    #
    # nu = tr.Float(0.2,
    #               label='nu',
    #               desc="Poison ratio",
    #               auto_set=False,
    #               input=True)

    # C40 SEQUENCE ORDER EFFECT

    #     gamma_T = Float(8000000.,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(100000.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.0024,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(6.2,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #     e_T = Float(12.,
    #                 label="e",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(3.6,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(1.7,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.008,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(100.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     Ad2 = Float(15.0,
    #                 label="a",
    #                 desc="brittleness coefficient",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     eps_0 = Float(0.0001,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #     eps_02 = Float(8e4,
    #                    label="a",
    #                    desc="threshold strain",
    #                    enter_set=True,
    #                    auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(100.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(100.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(50.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     state_var_shapes = Property(Dict(), depends_on='n_mp')
    #     '''Dictionary of state variable entries with their array shapes.
    #     '''
    #
    #     #-------------------------------------------------------------------------
    #     # Cached elasticity tensors
    #     #-------------------------------------------------------------------------
    #
    #     E = tr.Float(35e+3,
    #                  label="E",
    #                  desc="Young's Modulus",
    #                  auto_set=False,
    #                  input=True)
    #
    #     nu = tr.Float(0.2,
    #                   label='nu',
    #                   desc="Poison ratio",
    #                   auto_set=False,
    #                   input=True)

    #---------------------------------------
    # Tangential constitutive law parameters
    #     #---------------------------------------
    #     gamma_T = Float(8000000.,
    #                     label="Gamma",
    #                     desc=" Tangential Kinematic hardening modulus",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     K_T = Float(50000.0,
    #                 label="K",
    #                 desc="Tangential Isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     S_T = Float(0.01,
    #                 label="S",
    #                 desc="Damage strength",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     r_T = Float(3.,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     e_T = Float(4.,
    #                 label="r",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     c_T = Float(1.,
    #                 label="c",
    #                 desc="Damage cumulation parameter",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     tau_pi_bar = Float(1.,
    #                        label="Tau_bar",
    #                        desc="Reversibility limit",
    #                        enter_set=True,
    #                        auto_set=False)
    #
    #     a = Float(0.01,
    #               label="a",
    #               desc="Lateral pressure coefficient",
    #               enter_set=True,
    #               auto_set=False)
    #
    #     #-------------------------------------------
    #     # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #     #-------------------------------------------
    #     Ad = Float(500.0,
    #                label="a",
    #                desc="brittleness coefficient",
    #                enter_set=True,
    #                auto_set=False)
    #
    #     eps_0 = Float(0.00001,
    #                   label="a",
    #                   desc="threshold strain",
    #                   enter_set=True,
    #                   auto_set=False)
    #
    #     #-----------------------------------------------
    #     # Normal_Compression constitutive law parameters
    #     #-----------------------------------------------
    #     K_N = Float(1000.,
    #                 label="K_N",
    #                 desc=" Normal isotropic harening",
    #                 enter_set=True,
    #                 auto_set=False)
    #
    #     gamma_N = Float(100.,
    #                     label="gamma_N",
    #                     desc="Normal kinematic hardening",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     sigma_0 = Float(35.,
    #                     label="sigma_0",
    #                     desc="Yielding stress",
    #                     enter_set=True,
    #                     auto_set=False)
    #
    #     state_var_shapes = Property(Dict(), depends_on='n_mp')
    #     '''Dictionary of state variable entries with their array shapes.
    #     '''
    #
    #     #-------------------------------------------------------------------------
    #     # Cached elasticity tensors
    #     #-------------------------------------------------------------------------
    #
    #     E = tr.Float(35e+3,
    #                  label="E",
    #                  desc="Young's Modulus",
    #                  auto_set=False,
    #                  input=True)
    #
    #     nu = tr.Float(0.2,
    #                   label='nu',
    #                   desc="Poison ratio",
    #                   auto_set=False,
    #                   input=True)

    ###############
#     gamma_T = Float(10000.,
#                     label="Gamma",
#                     desc=" Tangential Kinematic hardening modulus",
#                     enter_set=True,
#                     auto_set=False)
#
#     K_T = Float(0000.0,
#                 label="K",
#                 desc="Tangential Isotropic harening",
#                 enter_set=True,
#                 auto_set=False)
#
#     S_T = Float(0.05,
#                 label="S",
#                 desc="Damage strength",
#                 enter_set=True,
#                 auto_set=False)
#
#     r_T = Float(1.,
#                 label="r",
#                 desc="Damage cumulation parameter",
#                 enter_set=True,
#                 auto_set=False)
#
#     e_T = Float(10.,
#                 label="r",
#                 desc="Damage cumulation parameter",
#                 enter_set=True,
#                 auto_set=False)
#
#     c_T = Float(1.,
#                 label="c",
#                 desc="Damage cumulation parameter",
#                 enter_set=True,
#                 auto_set=False)
#
#     tau_pi_bar = Float(.5,
#                        label="Tau_bar",
#                        desc="Reversibility limit",
#                        enter_set=True,
#                        auto_set=False)
#
#     a = Float(0.015,
#               label="a",
#               desc="Lateral pressure coefficient",
#               enter_set=True,
#               auto_set=False)
#
#     #-------------------------------------------
#     # Normal_Tension constitutive law parameters (without cumulative normal strain)
#     #-------------------------------------------
#     Ad = Float(500.0,
#                label="a",
#                desc="brittleness coefficient",
#                enter_set=True,
#                auto_set=False)
#
#     eps_0 = Float(0.00001,
#                   label="a",
#                   desc="threshold strain",
#                   enter_set=True,
#                   auto_set=False)
#
#     #-----------------------------------------------
#     # Normal_Compression constitutive law parameters
#     #-----------------------------------------------
#     K_N = Float(100.,
#                 label="K_N",
#                 desc=" Normal isotropic harening",
#                 enter_set=True,
#                 auto_set=False)
#
#     gamma_N = Float(100.,
#                     label="gamma_N",
#                     desc="Normal kinematic hardening",
#                     enter_set=True,
#                     auto_set=False)
#
#     sigma_0 = Float(10000.,
#                     label="sigma_0",
#                     desc="Yielding stress",
#                     enter_set=True,
#                     auto_set=False)
#
#     state_var_shapes = Property(Dict(), depends_on='n_mp')
#     '''Dictionary of state variable entries with their array shapes.
#     '''
#
#     #-------------------------------------------------------------------------
#     # Cached elasticity tensors
#     #-------------------------------------------------------------------------
#
#     E = tr.Float(35e+3,
#                  label="E",
#                  desc="Young's Modulus",
#                  auto_set=False,
#                  input=True)
#
#     nu = tr.Float(0.2,
#                   label='nu',
#                   desc="Poison ratio",
#                   auto_set=False,
#                   input=True)

#     gamma_T = Float(1000000.,
#                     label="Gamma",
#                     desc=" Tangential Kinematic hardening modulus",
#                     enter_set=True,
#                     auto_set=False)
#
#     K_T = Float(30000.0,
#                 label="K",
#                 desc="Tangential Isotropic harening",
#                 enter_set=True,
#                 auto_set=False)
#
#     S_T = Float(0.01,
#                 label="S",
#                 desc="Damage strength",
#                 enter_set=True,
#                 auto_set=False)
#
#     r_T = Float(14.,
#                 label="r",
#                 desc="Damage cumulation parameter",
#                 enter_set=True,
#                 auto_set=False)
#
#     c_T = Float(6.,
#                 label="c",
#                 desc="Damage cumulation parameter",
#                 enter_set=True,
#                 auto_set=False)
#
#     e_T = Float(14.,
#                 label="c",
#                 desc="Damage cumulation parameter",
#                 enter_set=True,
#                 auto_set=False)
#
#     tau_pi_bar = Float(2.0,
#                        label="Tau_bar",
#                        desc="Reversibility limit",
#                        enter_set=True,
#                        auto_set=False)
#
#     a = Float(0.01,
#               label="a",
#               desc="Lateral pressure coefficient",
#               enter_set=True,
#               auto_set=False)
#
#     #-------------------------------------------
#     # Normal_Tension constitutive law parameters (without cumulative normal strain)
#     #-------------------------------------------
#     Ad = Float(1000.0,
#                label="a",
#                desc="brittleness coefficient",
#                enter_set=True,
#                auto_set=False)
#
#     Ad2 = Float(1000.0,
#                 label="a",
#                 desc="brittleness coefficient",
#                 enter_set=True,
#                 auto_set=False)
#
#     eps_0 = Float(0.0001,
#                   label="a",
#                   desc="threshold strain",
#                   enter_set=True,
#                   auto_set=False)
#
#     eps_02 = Float(400.003,
#                    label="a",
#                    desc="threshold strain",
#                    enter_set=True,
#                    auto_set=False)
#
#     #-----------------------------------------------
#     # Normal_Compression constitutive law parameters
#     #-----------------------------------------------
#     K_N = Float(30000.,
#                 label="K_N",
#                 desc=" Normal isotropic harening",
#                 enter_set=True,
#                 auto_set=False)
#
#     gamma_N = Float(20000.,
#                     label="gamma_N",
#                     desc="Normal kinematic hardening",
#                     enter_set=True,
#                     auto_set=False)
#
#     sigma_0 = Float(60.,
#                     label="sigma_0",
#                     desc="Yielding stress",
#                     enter_set=True,
#                     auto_set=False)
#
#     state_var_shapes = Property(Dict(), depends_on='n_mp')
#     '''Dictionary of state variable entries with their array shapes.
#     '''
#
#     #-------------------------------------------------------------------------
#     # Cached elasticity tensors
#     #-------------------------------------------------------------------------
#
#     E = tr.Float(42e+3,
#                  label="E",
#                  desc="Young's Modulus",
#                  auto_set=False,
#                  input=True)
#
#     nu = tr.Float(0.2,
#                   label='nu',
#                   desc="Poison ratio",
#                   auto_set=False,
#                   input=True)

    def _get_lame_params(self):
        la = self.E * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E / (2. + 2. * self.nu)
        return la, mu

    D_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_abef(self):
        la = self._get_lame_params()[0]
        mu = self._get_lame_params()[1]
        delta = np.identity(2)
        D_abef = (np.einsum(',ij,kl->ijkl', la, delta, delta) +
                  np.einsum(',ik,jl->ijkl', mu, delta, delta) +
                  np.einsum(',il,jk->ijkl', mu, delta, delta))

        return D_abef

    @cached_property
    def _get_state_var_shapes(self):
        return {'w_N_Emn': (self.n_mp,),
                'z_N_Emn': (self.n_mp,),
                'alpha_N_Emn': (self.n_mp,),
                'r_N_Emn': (self.n_mp,),
                'eps_N_p_Emn': (self.n_mp,),
                'sigma_N_Emn': (self.n_mp,),
                'w_T_Emn': (self.n_mp,),
                'z_T_Emn': (self.n_mp,),
                'alpha_T_Emna': (self.n_mp, 2),
                'eps_T_pi_Emna': (self.n_mp, 2),
                }

    #--------------------------------------------------------------
    # microplane constitutive law (normal behavior CP + TD)
    # (without cumulative normal strain for fatigue under tension)
    #--------------------------------------------------------------
    def get_normal_law(self, eps_N_Emn, w_N_Emn, z_N_Emn,
                       alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux):

        eps_N_Aux = self._get_e_N_Emn_2(eps_aux)

        E_N = self.E / (1.0 - 2.0 * self.nu)

        sigma_trial = E_N * (eps_N_Emn - eps_N_p_Emn)
        pos = eps_N_Emn > 1e-6
        pos2 = eps_N_Emn < -1e-6
        H = 1.0 * pos
        H2 = 1.0 * pos2

        sigma_n_trial = (1.0 - H * w_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)
        sigma_N_Emn_tilde = E_N * (eps_N_Emn - eps_N_p_Emn)
        Z = self.K_N * r_N_Emn
        X = self.gamma_N * alpha_N_Emn * H2
        h = (self.sigma_0 + Z) * H2

        f_trial = (abs(sigma_N_Emn_tilde - X) - h) * H2

        thres_1 = f_trial > 1e-6

        delta_lamda = f_trial / \
            (E_N / (1 - w_N_Emn) + abs(self.K_N) + self.gamma_N) * thres_1
        eps_N_p_Emn = eps_N_p_Emn + delta_lamda * \
            np.sign(sigma_N_Emn_tilde - X)
        r_N_Emn = r_N_Emn + delta_lamda
        alpha_N_Emn = alpha_N_Emn + delta_lamda * \
            np.sign(sigma_N_Emn_tilde - X)

        def Z_N(z_N_Emn): return (1.0 / self.Ad) * (-z_N_Emn / (1.0 + z_N_Emn))

        Y_N = 0.5 * H * E_N * (eps_N_Emn - eps_N_p_Emn) ** 2.0
        Y_0 = 0.5 * E_N * self.eps_0 ** 2.0

        f = (Y_N - (Y_0 + Z_N(z_N_Emn)))

        thres_2 = f > 1e-6

        def f_w(Y): return 1.0 - 1.0 / (1.0 + self.Ad * (Y - Y_0))

        w_N_Emn[f > 1e-6] = f_w(Y_N)[f > 1e-6]
        z_N_Emn[f > 1e-6] = -w_N_Emn[f > 1e-6]

        sigma_N_Emn = (1.0 - H * w_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)
        sigma_N_Emn_tilde = E_N * (eps_N_Emn - eps_N_p_Emn)

        # pos2 = sigma_N_Emn * (eps_N_Emn - eps_N_Aux) > -1e-6
        # H2 = 1.0 * pos2
        # sigma_N_Emn = (1.0 - H * w_N_Emn) * E_N * (eps_N_Emn) * H2

        # WITH COMPRESSIVE DAMAGE

        #         E_N = self.E / (1.0 - 2.0 * self.nu)
        #
        #         sigma_trial = E_N * (eps_N_Emn - eps_N_p_Emn)
        #         pos = eps_N_Emn > 1e-6
        #         pos2 = eps_N_Emn < -1e-6
        #         H = 1.0 * pos
        #         H2 = 1.0 * pos2
        #
        #         sigma_n_trial = (1.0 - w_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)
        #         sigma_N_Emn_tilde = E_N * (eps_N_Emn - eps_N_p_Emn)
        #         Z = self.K_N * r_N_Emn
        #         X = self.gamma_N * alpha_N_Emn * H2
        #         h = (self.sigma_0 + Z) * H2
        #
        #         f_trial = (abs(sigma_N_Emn_tilde - X) - h) * H2
        #
        #         thres_1 = f_trial > 1e-6
        #
        #         delta_lamda = f_trial / \
        #             (E_N / (1 - w_N_Emn) + abs(self.K_N) + self.gamma_N) * thres_1
        #         eps_N_p_Emn = eps_N_p_Emn + delta_lamda * \
        #             np.sign(sigma_N_Emn_tilde - X)
        #         r_N_Emn = r_N_Emn + delta_lamda
        #         alpha_N_Emn = alpha_N_Emn + delta_lamda * \
        #             np.sign(sigma_N_Emn_tilde - X)
        #
        #         def Z_N(z_N_Emn): return 1.0 / self.Ad * (-z_N_Emn) / (1.0 + z_N_Emn)
        #
        #         def Z_N2(z_N_Emn): return 1.0 / self.Ad2 * (-z_N_Emn) / (1.0 + z_N_Emn)
        #
        #         Y_N = 0.5 * H * E_N * (eps_N_Emn - eps_N_p_Emn) ** 2.0
        #         Y_0 = 0.5 * E_N * self.eps_0 ** 2.0
        #
        #         Y_N2 = 0.5 * H2 * E_N * (eps_N_Emn - eps_N_p_Emn) ** 2.0
        #         Y_02 = 0.5 * E_N * self.eps_02 ** 2.0
        #
        #         f = (Y_N - (Y_0 + Z_N(z_N_Emn)))
        #
        #         thres_2 = f > 1e-6
        #
        #         def f_w(Y): return 1.0 - 1.0 / (1.0 + self.Ad * (Y - Y_0))
        #
        #         w_N_Emn[f > 1e-6] = f_w(Y_N)[f > 1e-6]
        #         z_N_Emn[f > 1e-6] = -w_N_Emn[f > 1e-6]
        #
        #         f2 = (Y_N2 - (Y_02 + Z_N2(z_N_Emn)))
        #
        #         thres_3 = f2 > 1e-6
        #
        #         def f_w2(Y): return 1.0 - 1.0 / (1.0 + self.Ad2 * (Y - Y_02))
        #
        #         w_N_Emn[f2 > 1e-6] = f_w2(Y_N2)[f2 > 1e-6]
        #         z_N_Emn[f2 > 1e-6] = -w_N_Emn[f2 > 1e-6]
        #
        #         sigma_N_Emn = (1.0 - w_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)
        #         sigma_N_Emn_tilde = E_N * (eps_N_Emn - eps_N_p_Emn)

        return w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Y_N, X



    #-------------------------------------------------------------------------
    # microplane constitutive law (Tangential CSD)-(Pressure sensitive cumulative damage)
    #-------------------------------------------------------------------------
    def get_tangential_law(self, eps_T_Emna, w_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, omegaN, sigma_kk):
        # E_T = self.E / (1.0 + self.nu)

        E_T = self.E * (1.0 - 4 * self.nu) / \
            ((1.0 + self.nu) * (1.0 - 2 * self.nu))

        sig_pi_trial = E_T * (eps_T_Emna - eps_T_pi_Emna)

        Z = self.K_T * z_T_Emn
        X = self.gamma_T * alpha_T_Emna
        norm_1 = np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))
        )
        Y = 0.5 * E_T * \
            np.einsum(
                '...na,...na->...n',
                (eps_T_Emna - eps_T_pi_Emna),
                (eps_T_Emna - eps_T_pi_Emna))

        f = norm_1 - self.tau_pi_bar - \
            Z + self.a * sigma_N_Emn

        plas_1 = f > 1e-6
        elas_1 = f < 1e-6

        delta_lamda = f / \
            (E_T / (1.0 - w_T_Emn) + self.gamma_T + self.K_T) * plas_1

        norm_2 = 1.0 * elas_1 + np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1

        eps_T_pi_Emna[..., 0] = eps_T_pi_Emna[..., 0] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 0] - X[..., 0]) /
             (1.0 - w_T_Emn)) / norm_2
        eps_T_pi_Emna[..., 1] = eps_T_pi_Emna[..., 1] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 1] - X[..., 1]) /
             (1.0 - w_T_Emn)) / norm_2

        w_T_Emn += ((1 - w_T_Emn) ** self.c_T) * \
            (delta_lamda * (Y / self.S_T) ** self.r_T) * \
            (self.tau_pi_bar / (self.tau_pi_bar + self.a * sigma_N_Emn)) ** self.e_T

        alpha_T_Emna[..., 0] = alpha_T_Emna[..., 0] + plas_1 * delta_lamda * \
            (sig_pi_trial[..., 0] - X[..., 0]) / norm_2
        alpha_T_Emna[..., 1] = alpha_T_Emna[..., 1] + plas_1 * delta_lamda * \
            (sig_pi_trial[..., 1] - X[..., 1]) / norm_2

        z_T_Emn = z_T_Emn + delta_lamda

        sigma_T_Emna = np.einsum(
            '...n,...na->...na', (1 - w_T_Emn), E_T * (eps_T_Emna - eps_T_pi_Emna))
        w_T_Emna = w_T_Emn

        return w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Y, X, w_T_Emna



#     #-------------------------------------------------------------------------
#     # MICROPLANE-Kinematic constraints
#     #-------------------------------------------------------------------------

    #-------------------------------------------------

    # get the operator of the microplane normals
    _MPNN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPNN(self):
        MPNN_nij = np.einsum('ni,nj->nij', self._MPN, self._MPN)
        return MPNN_nij

    # get the third order tangential tensor (operator) for each microplane
    _MPTT = Property(depends_on='n_mp')

    @cached_property
    def _get__MPTT(self):
        delta = np.identity(2)
        MPTT_nijr = 0.5 * (
            np.einsum('ni,jr -> nijr', self._MPN, delta) +
            np.einsum('nj,ir -> njir', self._MPN, delta) - 2 *
            np.einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN)
        )
        return MPTT_nijr

    def _get_e_N_Emn_2(self, eps_Emab):
        # get the normal strain array for each microplane
        return np.einsum('nij,...ij->...n', self._MPNN, eps_Emab)

    def _get_e_T_Emnar_2(self, eps_Emab):
        # get the tangential strain vector array for each microplane
        MPTT_ijr = self._get__MPTT()
        return np.einsum('nija,...ij->...na', MPTT_ijr, eps_Emab)

    #--------------------------------------------------------
    # return the state variables (Damage , inelastic strains)
    #--------------------------------------------------------
    def _get_state_variables(self, eps_Emab: object, tn1: object,
                             omegaN: object, z_N_Emn: object,
                             alpha_N_Emn: object, r_N_Emn: object, eps_N_p_Emn: object, sigma_N_Emn: object,
                             w_T_Emn: object, z_T_Emn: object, alpha_T_Emna: object, eps_T_pi_Emna: object, eps_aux: object, F: object) -> object:

        e_N_arr = self._get_e_N_Emn_2(eps_Emab)
        e_T_vct_arr = self._get_e_T_Emnar_2(eps_Emab)
        sigma_kk = np.abs(np.sum(F))


        omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Y_n, R_n = self.get_normal_law(e_N_arr,  omegaN, z_N_Emn,
                                                                                                        alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux)

        w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Y_T, X_T, w_T_Emna = self.get_tangential_law(e_T_vct_arr, w_T_Emn, z_T_Emn,alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, omegaN, sigma_kk)

        #D = np.sum(np.einsum('...n,...n->...n',w_T_Emn-w_T_Emn_aux,Y_T) + np.einsum('...n,...n->...n',omegaN_aux-omegaN,Y_n))
        #D = np.sum(np.einsum('...n,...n->...n',w_T_Emn-w_T_Emn_aux,Y_T) + np.einsum('...n,...n->...n',omegaN_aux-omegaN,Y_n) + np.einsum('...n,...n->...',eps_T_pi_Emna_aux-eps_T_pi_Emna,sigma_T_Emna) + np.einsum('...n,...n->...n',eps_N_p_Emn_aux-eps_N_p_Emn,sigma_N_Emn))
        #D = np.sum(np.einsum('...n,...n->...n',self._MPW,np.einsum('...n,...n->...n',w_T_Emn,Y_T)) + np.einsum('...n,...n->...n',self._MPW,np.einsum('...n,...n->...n',omegaN,Y_n)) + np.einsum('...n,...n->...n',self._MPW,np.einsum('...n,...n->...',eps_T_pi_Emna,sigma_T_Emna)) + np.einsum('...n,...n->...n',self._MPW,np.einsum('...n,...n->...n',eps_N_p_Emn,sigma_N_Emn)))

#         print(eps_N_p_Emn[0], 'eps p')
#         print(Y_n[0], 'lambda')
        return omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Y_n, R_n, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Y_T, X_T

    #---------------------------------------------------------------------
    # Extra homogenization of damage tensor in case of two damage parameters
    # Returns the 4th order damage tensor 'beta4' using (ref. [Baz99], Eq.(63))
    #---------------------------------------------------------------------

    def _get_beta_Emabcd_2(self, eps_Emab, w_N_Emn, z_N_Emn,
                           alpha_N_Emn, r_N_Emn, eps_N_p_Emn, w_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, eps_aux, sigma_kk):

        # Returns the 4th order damage tensor 'beta4' using
        #(cf. [Baz99], Eq.(63))

        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)

        w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Y_n, R_n = self.get_normal_law(
            eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux)

        w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Y_T, X_T, w_T_Emna = self.get_tangential_law(
            eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, w_N_Emn, sigma_kk)

        delta = np.identity(2)
        beta_N = np.sqrt(1. - w_N_Emn)
        beta_T = np.sqrt(1. - w_T_Emn)

        beta_ijkl = np.einsum('n, ...n,ni, nj, nk, nl -> ...ijkl', self._MPW, beta_N, self._MPN, self._MPN, self._MPN, self._MPN) + \
            0.25 * (np.einsum('n, ...n,ni, nk, jl -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n, ...n,ni, nl, jk -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n, ...n,nj, nk, il -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n, ...n,nj, nl, ik -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) -
                    4.0 * np.einsum('n, ...n, ni, nj, nk, nl -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, self._MPN, self._MPN))

        return beta_ijkl
    #-----------------------------------------------------------
    # Integration of the (inelastic) strains for each microplane
    #-----------------------------------------------------------

    def _get_eps_p_Emab(self, eps_Emab, w_N_Emn, z_N_Emn,
                        alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                        w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, eps_aux, sigma_kk):

        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)

        # plastic normal strains
        w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Y_n, R_n = self.get_normal_law(
            eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux)

        # sliding tangential strains
        w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Y_T, X_T, w_T_Emna = self.get_tangential_law(
            eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, w_N_Emn, sigma_kk)

        delta = np.identity(2)

        # 2-nd order plastic (inelastic) tensor
        eps_p_Emab = (
            np.einsum('n,...n,na,nb->...ab',
                      self._MPW, eps_N_p_Emn, self._MPN, self._MPN) +
            0.5 * (
                np.einsum('n,...nf,na,fb->...ab',
                          self._MPW, eps_T_pi_Emna, self._MPN, delta) +
                np.einsum('n,...nf,nb,fa->...ab', self._MPW,
                          eps_T_pi_Emna, self._MPN, delta)
            )
        )

        return eps_p_Emab

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, eps_Emab, t_n1, w_N_Emn, z_N_Emn,
                      alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                      w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, eps_aux, F):

        # Corrector predictor computation.

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------

        sigma_kk = np.sum(F)

        beta_Emabcd = self._get_beta_Emabcd_2(
            eps_Emab, w_N_Emn, z_N_Emn,
            alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
            w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, eps_aux, sigma_kk
        )

        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------

        D_Emabcd = np.einsum(
            '...ijab, abef, ...cdef->...ijcd', beta_Emabcd, self.D_abef, beta_Emabcd)

        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------
        # plastic strain tensor
        eps_p_Emab = self._get_eps_p_Emab(
            eps_Emab, w_N_Emn, z_N_Emn,
            alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
            w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, eps_aux, sigma_kk)

        # elastic strain tensor
        eps_e_Emab = eps_Emab - eps_p_Emab

        # calculation of the stress tensor
        sig_Emab = np.einsum('...abcd,...cd->...ab', D_Emabcd, eps_e_Emab)

        return D_Emabcd, sig_Emab, eps_p_Emab

# class MATS2DMplCSDEEQ(MATSXDMplCDSEEQ, MATS2DEval):

    # implements(IMATSEval)

    #-----------------------------------------------
    # number of microplanes
    #-----------------------------------------------
    n_mp = Constant(100)

    #-----------------------------------------------
    # get the normal vectors of the microplanes
    #-----------------------------------------------
    _MPN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPN(self):
        # microplane normals:
        alpha_list = np.linspace(0, 2 * np.pi, self.n_mp)

        MPN = np.array([[np.cos(alpha), np.sin(alpha)]
                        for alpha in alpha_list])

        return MPN

    #-------------------------------------
    # get the weights of the microplanes
    #-------------------------------------
    _MPW = Property(depends_on='n_mp')

    @cached_property
    def _get__MPW(self):
        MPW = np.ones(self.n_mp) / self.n_mp * 2

        return MPW


if __name__ == '__main__':
    #==========================================================================
    # Check the model behavior at the single material point
    #==========================================================================
    model = MATS2DMplCSDEEQ()
    p = 1.0  # ratio of strain eps_11 (for bi-axial loading)
    m = 0.

    n_cycles = 1
    T = 1 / n_cycles
    eps_max = -0.01
    t_steps = 500 * n_cycles

    t = np.linspace(0, 1, t_steps)

    sin_load = np.linspace(0, eps_max, t_steps)

    eps_ab = np.array([np.array([[p * sin_load[i], 0],
                                 [0,  -m * sin_load[i]]]) for i in range(0, len(sin_load))])

    t_steps_total = len(sin_load)

    n_mp = 100
    omegaN = np.zeros((n_mp, ))
    z_N_Emn = np.zeros((n_mp, ))
    alpha_N_Emn = np.zeros((n_mp, ))
    r_N_Emn = np.zeros((n_mp, ))
    eps_N_p_Emn = np.zeros((n_mp, ))
    sigma_N_Emn = np.zeros((n_mp, ))
    Y_n = np.zeros((n_mp, ))
    R_n = np.zeros((n_mp, ))
    w_T_Emn = np.zeros((n_mp, ))
    z_T_Emn = np.zeros((n_mp, ))
    alpha_T_Emna = np.zeros((n_mp, 2))
    eps_T_pi_Emna = np.zeros((n_mp, 2))
    sigma_T_Emna = np.zeros((n_mp, 2))
    X_T = np.zeros((n_mp, 2))
    Y_T = np.zeros((n_mp, ))
    eps_aux = np.zeros((2, 2))
    F_O = np.zeros((3,))
    sigma_t = np.zeros((2, 2), np.float_)
    sigma_t = sigma_t[np.newaxis, :, :]

    for i in range(t_steps_total):
        D_abcd, sig_ab = model.get_corr_pred(
            eps_ab[i], 1, omegaN, z_N_Emn,
            alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
            w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, eps_aux, F_O
        )
        sigma_aux = sig_ab[np.newaxis, :, :]
        sigma_t = np.concatenate((sigma_t, sigma_aux))

        [omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Y_n, R_n, w_T_Emn, z_T_Emn,
            alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Y_T, X_T] = model._get_state_variables(
                eps_ab[i], 1, omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, eps_aux, F_O)

        omegaN = omegaN.reshape(n_mp, )
        z_N_Emn = z_N_Emn.reshape(n_mp, )
        alpha_N_Emn = alpha_N_Emn.reshape(n_mp, )
        r_N_Emn = r_N_Emn.reshape(n_mp, )
        eps_N_p_Emn = eps_N_p_Emn.reshape(n_mp, )
        sigma_N_Emn = sigma_N_Emn.reshape(n_mp,)
        Y_n = Y_n.reshape(n_mp,)
        R_n = R_n.reshape(n_mp,)
        w_T_Emn = w_T_Emn.reshape(n_mp, )
        z_T_Emn = z_T_Emn.reshape(n_mp, )
        alpha_T_Emna = alpha_T_Emna.reshape(n_mp, 2)
        eps_T_pi_Emna = eps_T_pi_Emna.reshape(n_mp, 2)
        sigma_T_Emna = sigma_T_Emna.reshape(n_mp, 2)
        X_T = X_T.reshape(n_mp, 2)
        Y_T = Y_T.reshape(n_mp, )

    import matplotlib.pyplot as plt
    print(sigma_t.shape)

    f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))

    ax2.plot(np.abs(eps_ab[:, 0, 0]), np.abs(
        sigma_t[1:, 0, 0]), 'k', linewidth=3.5)
    ax2.set_xlabel(r'$|\varepsilon_{11}$| [-]', fontsize=25)
    ax2.set_ylabel(r'$|\sigma{11}$| [-]', fontsize=25)
    #ax2.set_xlim(-0.00001, 0.01)

    # ax2.set_ylim(-0.00001, 70)
    plt.show()
