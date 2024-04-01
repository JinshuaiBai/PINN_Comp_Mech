def Material(U_x, U_y, U_z, V_x, V_y, V_z, W_x, W_y, W_z,
             U_xx, U_xy, U_xz, U_yy, U_yz, U_zz,
             V_xx, V_xy, V_xz, V_yy, V_yz, V_zz,
             W_xx, W_xy, W_xz, W_yy, W_yz, W_zz, E, mu):
    """
    ====================================================================================================================

    This function is to calculate the strain and stress based on the constitutive equation.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [U_x]       [Keras tensor]          : First-order derivative of displacement u with respect to x direction;
    [U_y]       [Keras tensor]          : First-order derivative of displacement u with respect to y direction;
    [U_z]       [Keras tensor]          : First-order derivative of displacement u with respect to z direction;
    [U_xx]      [Keras tensor]          : Second-order derivative of displacement u with respect to x direction;
    [U_xy]      [Keras tensor]          : Second-order derivative of displacement u with respect to x and y directions;
    [U_xz]      [Keras tensor]          : Second-order derivative of displacement u with respect to x and z directions;
    [U_yy]      [Keras tensor]          : Second-order derivative of displacement u with respect to y direction;
    [U_yz]      [Keras tensor]          : Second-order derivative of displacement u with respect to y and z directions;
    [U_zz]      [Keras tensor]          : Second-order derivative of displacement u with respect to z direction;
    [V_x]       [Keras tensor]          : First-order derivative of displacement v with respect to x direction;
    [V_y]       [Keras tensor]          : First-order derivative of displacement v with respect to y direction;
    [V_z]       [Keras tensor]          : First-order derivative of displacement v with respect to z direction;
    [V_xx]      [Keras tensor]          : Second-order derivative of displacement v with respect to x direction;
    [V_xy]      [Keras tensor]          : Second-order derivative of displacement v with respect to x and y directions;
    [V_xz]      [Keras tensor]          : Second-order derivative of displacement v with respect to x and z directions;
    [V_yy]      [Keras tensor]          : Second-order derivative of displacement v with respect to y direction;
    [V_yz]      [Keras tensor]          : Second-order derivative of displacement v with respect to y and z directions;
    [V_zz]      [Keras tensor]          : Second-order derivative of displacement v with respect to z direction;
    [W_x]       [Keras tensor]          : First-order derivative of displacement w with respect to x direction;
    [W_y]       [Keras tensor]          : First-order derivative of displacement w with respect to y direction;
    [W_z]       [Keras tensor]          : First-order derivative of displacement w with respect to z direction;
    [W_xx]      [Keras tensor]          : Second-order derivative of displacement w with respect to x direction;
    [W_xy]      [Keras tensor]          : Second-order derivative of displacement w with respect to x and y directions;
    [W_xz]      [Keras tensor]          : Second-order derivative of displacement w with respect to x and z directions;
    [W_yy]      [Keras tensor]          : Second-order derivative of displacement w with respect to y direction;
    [W_yz]      [Keras tensor]          : Second-order derivative of displacement w with respect to y and z directions;
    [W_zz]      [Keras tensor]          : Second-order derivative of displacement w with respect to z direction;
    [e1]        [Keras tensor]          : Normal strain for x direction;
    [e2]        [Keras tensor]          : Normal strain for y direction;
    [e3]        [Keras tensor]          : Normal strain for z direction;
    [e12]       [Keras tensor]          : Shear strain for x and y directions;
    [e13]       [Keras tensor]          : Shear strain for x and z directions;
    [e23]       [Keras tensor]          : Shear strain for y and z directions;
    [s1]        [Keras tensor]          : Normal stress for x direction;
    [s2]        [Keras tensor]          : Normal stress for y direction;
    [s2]        [Keras tensor]          : Normal stress for z direction;
    [s12]       [Keras tensor]          : Shear stress for x and y directions;
    [s13]       [Keras tensor]          : Shear stress for x and z directions;
    [s23]       [Keras tensor]          : Shear stress for y and z directions;
    [Gex]       [Keras tensor]          : Residual from the equilibrium equation for x direction;
    [Gey]       [Keras tensor]          : Residual from the equilibrium equation for y direction;
    [Gez]       [Keras tensor]          : Residual from the equilibrium equation for z direction;
    [E]         [float]                 : Young's module;
    [mu]        [float]                 : Poisson ratio;
    [la]        [float]                 : First lame constant;
    [nu]        [float]                 : Second lame constant.

    ====================================================================================================================
    """
    ### Calculate the Lame constants
    la = E * mu/(1 + mu) / (1 - 2 * mu)
    G = E / (1 + mu) / 2
    
    ### Calculate strain
    e1 = U_x
    e2 = V_y
    e3 = W_z
    e12 = 0.5 * (U_y + V_x)
    e23 = 0.5 * (W_y + V_z)
    e13 = 0.5 * (U_z + W_x)

    ### Calculate stress
    s1 = (2 * G + la) * e1 + la * e2 + la * e3
    s2 = (2 * G + la) * e2 + la * e1 + la * e3
    s3 = (2 * G + la) * e3 + la * e1 + la * e2
    s12 = 2 * G * e12
    s23 = 2 * G * e23
    s13 = 2 * G * e13

    ### Calculate the residual from equilibrium equation
    Gex = (G + la) * (U_xx + V_xy + W_xz) + G * (U_xx + U_yy + U_zz)
    Gey = (G + la) * (V_yy + U_xy + W_yz) + G * (V_xx + V_yy + V_zz)
    Gez = (G + la) * (W_zz + U_xz + V_yz) + G * (W_xx + W_yy + W_zz)

    return e1, e2, e3, e12, e23, e13, s1, s2, s3, s12, s23, s13, Gex, Gey, Gez
