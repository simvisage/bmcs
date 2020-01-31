def construct_fail_envelope():

    elastic_debug = False
    # Tseval for a material model
    #
    tseval  = MATS2DMicroplaneDamage( elastic_debug = elastic_debug )


    value, coeff = get_value_and_coeff( 1., 0.0 )

    bcond_alpha = BCDof(var='u', dof = 0, value = value,
                     link_dofs = [1],
                     link_coeffs = [coeff],
                     time_function = lambda t: t )
    
    ts = TS( tse = tseval,
             bcond_list = [ self.bcond_alpha 
                         ],
             rtrace_list = [ RTraceGraph(name = 'strain 0 - stress 0',
                                  var_x = 'eps_app', idx_x = 0,
                                  var_y = 'sig_app', idx_y = 0,
                                  update_on = 'update' ),
                         RTraceGraph(name = 'strain 1 - stress 1',
                                  var_x = 'eps_app', idx_x = 1,
                                  var_y = 'sig_app', idx_y = 1,
                                  update_on = 'update' ),
                         RTraceGraph(name = 'strain 0 - stress 1',
                                  var_x = 'eps_app', idx_x = 0,
                                  var_y = 'sig_app', idx_y = 1,
                                  update_on = 'update' ),
                         RTraceGraph(name = 'strain 1 - stress 0',
                                  var_x = 'eps_app', idx_x = 1,
                                  var_y = 'sig_app', idx_y = 0,
                                  update_on = 'update' ),
                         RTraceGraph(name = 'strain 0 - strain 1',
                                  var_x = 'eps_app', idx_x = 0,
                                  var_y = 'eps_app', idx_y = 1,
                                  update_on = 'update' ),
                         ]
                         )

    # Put the time-stepper into the time-loop
    #
    if elastic_debug:
        tmax = 1.
        n_steps = 1
    else:
        tmax = 0.001
        # tmax = 0.0006
        n_steps = 100

    tl = TL( ts = ts,
             DT=tmax/n_steps, KMAX = 100, RESETMAX = 0,
             T = TRange( min = 0.0,  max = tmax ) )

    from numpy import argmax

    alpha_arr = linspace( - Pi/2 * 1.05,  2*(Pi/2.) + Pi/2.*0.05, 20 )

    sig0_m_list = []
    sig1_m_list = []
    eps0_m_list = []
    eps1_m_list = []

    for alpha in alpha_arr:
    
        value, coeff = get_value_and_coeff( 1., alpha )
        bcond_alpha.value = value
        bcond_alpha.link_coeffs[0] = coeff

        tl.eval()

        eps0_sig0 = tl.rv_mngr.rv_list[0]
        eps1_sig1 = tl.rv_mngr.rv_list[1]

        sig0_midx = argmax( fabs( eps0_sig0.trace.ydata ) )
        sig1_midx = argmax( fabs( eps1_sig1.trace.ydata ) )

        sig0_m = eps0_sig0.trace.ydata[ sig0_midx ]
        sig1_m = eps1_sig1.trace.ydata[ sig1_midx ]
        
        eps0_m = eps0_sig0.trace.xdata[ sig0_midx ]
        eps1_m = eps1_sig1.trace.xdata[ sig1_midx ]

        sig0_m_list.append( sig0_m )
        sig1_m_list.append( sig1_m )
        eps0_m_list.append( eps0_m )
        eps1_m_list.append( eps1_m )

    from math_func import MFnLineArray
    
    sig_plot = MFnLineArray( xdata = sig0_m_list,
                              ydata = sig1_m_list )
    eps_plot = MFnLineArray( xdata = eps0_m_list,
                              ydata = eps1_m_list )
    sig_plot.configure_traits()
    
    # Put the time-loop into the simulation-framework and map the
    # object to the user interface.
    #
