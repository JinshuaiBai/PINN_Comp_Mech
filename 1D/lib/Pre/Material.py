def Material(U_x, U_xx, E):
    """
    ====================================================================================================================

    This function is to calculate the strain and stress based on the constitutive equation.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [U_x]       [Keras tensor]          : First-order derivative of displacement with respect to x direction;
    [U_xx]      [Keras tensor]          : Second-order derivative of displacement with respect to x direction;
    [epsilon]   [Keras tensor]          : Strain;
    [sigma]     [Keras tensor]          : Stress;
    [Ge]        [Keras tensor]          : Residual from the equilibrium equation;
    [E]         [float]                 : Young's module.

    ====================================================================================================================
    """

    ### Calculate strain
    epsilon = U_x

    ### Calculate stress
    sigma = E * epsilon

    ### Calculate the residual from equilibrium equation
    Ge = E * U_xx

    return epsilon, sigma, Ge
