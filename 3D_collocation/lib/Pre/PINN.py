import tensorflow as tf
from lib.Pre.Material import Material
from lib.Pre.Dif_op_x import Dif_x
from lib.Pre.Dif_op_y import Dif_y
from lib.Pre.Dif_op_z import Dif_z

def PINN(net_u, net_v, net_w, E, mu):
    """
    ====================================================================================================================

    This function is to initialize a PINN.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [net_u]     [keras model]           : The trained FNN for displacement u;

    [s_u_x]     [Array of float32]      : x direction force boundary condition on the top boundary of the plate;
    [s_u_y]     [Array of float32]      : y direction force boundary condition on the top boundary of the plate;
    [s_b_x]     [Array of float32]      : x direction force boundary condition on the bottom boundary of the plate;
    [s_b_y]     [Array of float32]      : y direction force boundary condition on the bottom boundary of the plate;
    [s_l_x]     [Array of float32]      : x direction force boundary condition on the left boundary of the plate;
    [s_l_y]     [Array of float32]      : y direction force boundary condition on the left boundary of the plate;
    [s_r_x]     [Array of float32]      : x direction force boundary condition on the right boundary of the plate;
    [s_r_y]     [Array of float32]      : y direction force boundary condition on the right boundary of the plate;
    [Gex]       [Keras tensor]          : Residual from the equilibrium equation for x direction;
    [Gey]       [Keras tensor]          : Residual from the equilibrium equation for y direction;
    [sigma_r]   [Keras tensor]          : Stress at the right tip of the rod;
    [E]         [float]                 : Young's module;
    [mu]        [float]                 : Poisson ratio.

    ====================================================================================================================
    """

    ### Declare inputs
    x = tf.keras.layers.Input(shape=(3,))
    x1u = tf.keras.layers.Input(shape=(3,))
    x1b = tf.keras.layers.Input(shape=(3,))
    x2u = tf.keras.layers.Input(shape=(3,))
    x2b = tf.keras.layers.Input(shape=(3,))
    x3u = tf.keras.layers.Input(shape=(3,))
    x3b = tf.keras.layers.Input(shape=(3,))

    ### Initialize the differential operators
    dif_x = Dif_x(net_u)
    dif_y = Dif_y(net_v)
    dif_z = Dif_z(net_w)
    
    ### Obtain partial derivatives with respect to x and y
    U_x, U_y, U_z, U_xx, U_xy, U_xz, U_yy, U_yz, U_zz = dif_x(x)
    V_x, V_y, V_z, V_xx, V_xy, V_xz, V_yy, V_yz, V_zz = dif_y(x)
    W_x, W_y, W_z, W_xx, W_xy, W_xz, W_yy, W_yz, W_zz = dif_z(x)

    U_1u_x, U_1u_y, U_1u_z, U_1u_xx, U_1u_xy, U_1u_xz, U_1u_yy, U_1u_yz, U_1u_zz = dif_x(x1u)
    V_1u_x, V_1u_y, V_1u_z, V_1u_xx, V_1u_xy, V_1u_xz, V_1u_yy, V_1u_yz, V_1u_zz = dif_y(x1u)
    W_1u_x, W_1u_y, W_1u_z, W_1u_xx, W_1u_xy, W_1u_xz, W_1u_yy, W_1u_yz, W_1u_zz = dif_z(x1u)

    U_1b_x, U_1b_y, U_1b_z, U_1b_xx, U_1b_xy, U_1b_xz, U_1b_yy, U_1b_yz, U_1b_zz = dif_x(x1b)
    V_1b_x, V_1b_y, V_1b_z, V_1b_xx, V_1b_xy, V_1b_xz, V_1b_yy, V_1b_yz, V_1b_zz = dif_y(x1b)
    W_1b_x, W_1b_y, W_1b_z, W_1b_xx, W_1b_xy, W_1b_xz, W_1b_yy, W_1b_yz, W_1b_zz = dif_z(x1b)

    U_2u_x, U_2u_y, U_2u_z, U_2u_xx, U_2u_xy, U_2u_xz, U_2u_yy, U_2u_yz, U_2u_zz = dif_x(x2u)
    V_2u_x, V_2u_y, V_2u_z, V_2u_xx, V_2u_xy, V_2u_xz, V_2u_yy, V_2u_yz, V_2u_zz = dif_y(x2u)
    W_2u_x, W_2u_y, W_2u_z, W_2u_xx, W_2u_xy, W_2u_xz, W_2u_yy, W_2u_yz, W_2u_zz = dif_z(x2u)

    U_2b_x, U_2b_y, U_2b_z, U_2b_xx, U_2b_xy, U_2b_xz, U_2b_yy, U_2b_yz, U_2b_zz = dif_x(x2b)
    V_2b_x, V_2b_y, V_2b_z, V_2b_xx, V_2b_xy, V_2b_xz, V_2b_yy, V_2b_yz, V_2b_zz = dif_y(x2b)
    W_2b_x, W_2b_y, W_2b_z, W_2b_xx, W_2b_xy, W_2b_xz, W_2b_yy, W_2b_yz, W_2b_zz = dif_z(x2b)

    U_3u_x, U_3u_y, U_3u_z, U_3u_xx, U_3u_xy, U_3u_xz, U_3u_yy, U_3u_yz, U_3u_zz = dif_x(x3u)
    V_3u_x, V_3u_y, V_3u_z, V_3u_xx, V_3u_xy, V_3u_xz, V_3u_yy, V_3u_yz, V_3u_zz = dif_y(x3u)
    W_3u_x, W_3u_y, W_3u_z, W_3u_xx, W_3u_xy, W_3u_xz, W_3u_yy, W_3u_yz, W_3u_zz = dif_z(x3u)

    U_3b_x, U_3b_y, U_3b_z, U_3b_xx, U_3b_xy, U_3b_xz, U_3b_yy, U_3b_yz, U_3b_zz = dif_x(x3b)
    V_3b_x, V_3b_y, V_3b_z, V_3b_xx, V_3b_xy, V_3b_xz, V_3b_yy, V_3b_yz, V_3b_zz = dif_y(x3b)
    W_3b_x, W_3b_y, W_3b_z, W_3b_xx, W_3b_xy, W_3b_xz, W_3b_yy, W_3b_yz, W_3b_zz = dif_z(x3b)

    ### Obtain the residuals from stress boundary conditions
    _, _, _, _, _, _, _, _, _, _, _, _, Gex, Gey, Gez = Material(
        U_x, U_y, U_z, V_x, V_y, V_z, W_x, W_y, W_z,
        U_xx, U_xy, U_xz, U_yy, U_yz, U_zz,
        V_xx, V_xy, V_xz, V_yy, V_yz, V_zz,
        W_xx, W_xy, W_xz, W_yy, W_yz, W_zz, E, mu)

    _, _, _, _, _, _, s11u, _, _, s121u, _, s131u, _, _, _ = Material(
        U_1u_x, U_1u_y, U_1u_z, V_1u_x, V_1u_y, V_1u_z, W_1u_x, W_1u_y, W_1u_z,
        U_1u_xx, U_1u_xy, U_1u_xz, U_1u_yy, U_1u_yz, U_1u_zz,
        V_1u_xx, V_1u_xy, V_1u_xz, V_1u_yy, V_1u_yz, V_1u_zz,
        W_1u_xx, W_1u_xy, W_1u_xz, W_1u_yy, W_1u_yz, W_1u_zz, E, mu)

    _, _, _, _, _, _, _, _, _, s121b, _, s131b, _, _, _ = Material(
        U_1b_x, U_1b_y, U_1b_z, V_1b_x, V_1b_y, V_1b_z, W_1b_x, W_1b_y, W_1b_z,
        U_1b_xx, U_1b_xy, U_1b_xz, U_1b_yy, U_1b_yz, U_1b_zz,
        V_1b_xx, V_1b_xy, V_1b_xz, V_1b_yy, V_1b_yz, V_1b_zz,
        W_1b_xx, W_1b_xy, W_1b_xz, W_1b_yy, W_1b_yz, W_1b_zz, E, mu)

    _, _, _, _, _, _, _, s22u, _, s122u, s232u, _, _, _, _ = Material(
        U_2u_x, U_2u_y, U_2u_z, V_2u_x, V_2u_y, V_2u_z, W_2u_x, W_2u_y, W_2u_z,
        U_2u_xx, U_2u_xy, U_2u_xz, U_2u_yy, U_2u_yz, U_2u_zz,
        V_2u_xx, V_2u_xy, V_2u_xz, V_2u_yy, V_2u_yz, V_2u_zz,
        W_2u_xx, W_2u_xy, W_2u_xz, W_2u_yy, W_2u_yz, W_2u_zz, E, mu)

    _, _, _, _, _, _, _, _, _, s122b, s232b, _, _, _, _ = Material(
        U_2b_x, U_2b_y, U_2b_z, V_2b_x, V_2b_y, V_2b_z, W_2b_x, W_2b_y, W_2b_z,
        U_2b_xx, U_2b_xy, U_2b_xz, U_2b_yy, U_2b_yz, U_2b_zz,
        V_2b_xx, V_2b_xy, V_2b_xz, V_2b_yy, V_2b_yz, V_2b_zz,
        W_2b_xx, W_2b_xy, W_2b_xz, W_2b_yy, W_2b_yz, W_2b_zz, E, mu)

    _, _, _, _, _, _, _, _, s33u, _, s233u, s133u, _, _, _ = Material(
        U_3u_x, U_3u_y, U_3u_z, V_3u_x, V_3u_y, V_3u_z, W_3u_x, W_3u_y, W_3u_z,
        U_3u_xx, U_3u_xy, U_3u_xz, U_3u_yy, U_3u_yz, U_3u_zz,
        V_3u_xx, V_3u_xy, V_3u_xz, V_3u_yy, V_3u_yz, V_3u_zz,
        W_3u_xx, W_3u_xy, W_3u_xz, W_3u_yy, W_3u_yz, W_3u_zz, E, mu)

    _, _, _, _, _, _, _, _, _, _, s233b, s133b, _, _, _ = Material(
        U_3b_x, U_3b_y, U_3b_z, V_3b_x, V_3b_y, V_3b_z, W_3b_x, W_3b_y, W_3b_z,
        U_3b_xx, U_3b_xy, U_3b_xz, U_3b_yy, U_3b_yz, U_3b_zz,
        V_3b_xx, V_3b_xy, V_3b_xz, V_3b_yy, V_3b_yz, V_3b_zz,
        W_3b_xx, W_3b_xy, W_3b_xz, W_3b_yy, W_3b_yz, W_3b_zz, E, mu)

    ### build up the PINN
    pinn = tf.keras.models.Model(inputs=[x, x1u, x1b, x2u, x2b, x3u, x3b],
        outputs=[Gex, Gey, Gez,
            s11u, s121u, s131u, s121b, s131b,
            s22u, s122u, s232u, s122b, s232b,
            s33u, s133u, s233u, s133b, s233b])

    return pinn