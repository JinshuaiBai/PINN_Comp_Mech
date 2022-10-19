def Material(U_x, U_y, V_x, V_y, U_xx, U_xy, U_yy, V_xx, V_xy, V_yy, E, mu, p):
    """
    ====================================================================================================================

    This function is to calculate the strain and stress based on the constitutive equation.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [U_x]       [Keras tensor]          : First-order derivative of displacement u with respect to x direction;
    [U_y]       [Keras tensor]          : First-order derivative of displacement u with respect to y direction;
    [U_xx]      [Keras tensor]          : Second-order derivative of displacement u with respect to x direction;
    [U_xy]      [Keras tensor]          : Second-order derivative of displacement u with respect to x and y directions;
    [U_yy]      [Keras tensor]          : Second-order derivative of displacement u with respect to y direction;
    [V_x]       [Keras tensor]          : First-order derivative of displacement u with respect to x direction;
    [V_y]       [Keras tensor]          : First-order derivative of displacement u with respect to y direction;
    [V_xx]      [Keras tensor]          : Second-order derivative of displacement u with respect to x direction;
    [V_xy]      [Keras tensor]          : Second-order derivative of displacement u with respect to x and y directions;
    [V_yy]      [Keras tensor]          : Second-order derivative of displacement u with respect to y direction;
    [e1]        [Keras tensor]          : Normal strain for x direction;
    [e2]        [Keras tensor]          : Normal strain for y direction;
    [e12]       [Keras tensor]          : Shear strain;
    [s1]        [Keras tensor]          : Normal stress for x direction;
    [s2]        [Keras tensor]          : Normal stress for y direction;
    [s12]       [Keras tensor]          : Shear stress;
    [Gex]       [Keras tensor]          : Residual from the equilibrium equation for x direction;
    [Gey]       [Keras tensor]          : Residual from the equilibrium equation for y direction;
    [E]         [float]                 : Young's module;
    [mu]        [float]                 : Poisson ratio.

    ====================================================================================================================
    """
    
    if p == 'plain_strain':
        ### plain strain
        la = E * mu/(1 + mu) / (1 - 2 * mu)
        nu = E / (1 + mu) / 2 
    elif p == 'plain_stress':
        ### plain stress
        la = E * mu/(1 + mu) / (1 - mu)
        nu = E / (1 + mu) / 2
    else:
        print('-------------------------------------------------\n')
        print('Material property error!\n')
        print('-------------------------------------------------\n')
        print('Please select the one of the following options:\n1.\tplain_strain\n2.\tplain_stress\n')
        print('-------------------------------------------------\n')
    
    ### Calculate strain
    e1 = U_x
    e2 = V_y
    e12 = 0.5 * (U_y + V_x)
    
    ### Calculate stress
    s1 = (2 * nu + la) * e1 + la * e2
    s2 = (2 * nu + la) * e2 + la * e1
    s12 = 2 * nu * e12
    
    ### Calculate the residual from equilibrium equation
    Gex = (2 * nu + la) * U_xx + nu * U_yy + (nu + la) * V_xy
    Gey = (nu + la) * U_xy + (2 * nu + la) * V_yy + nu * V_xx
    
    return e1, e2, e12, s1, s2, s12, Gex, Gey
    