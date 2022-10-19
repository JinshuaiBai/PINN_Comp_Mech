import tensorflow as tf
from lib.Pre.Material import Material
from lib.Pre.Dif_op_x import Dif_x
from lib.Pre.Dif_op_y import Dif_y

def PINN(net_u, net_v, E, mu):
    """
    ====================================================================================================================

    This function is to initialize a PINN.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [net_u]     [keras model]           : The trained FNN for displacement u;
    [xy]        [Array of float32]      : Coordinates of all the sample points;
    [xy_u]      [Array of float32]      : Coordinates of the sample points on the upper boundary of the plate;
    [xy_b]      [Array of float32]      : Coordinates of the sample points on the bottom boundary of the plate;
    [xy_l]      [Array of float32]      : Coordinates of the sample points on the left boundary of the plate;
    [xy_r]      [Array of float32]      : Coordinates of the sample points on the right boundary of the plate;

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

    ### declare PINN's inputs
    xy = tf.keras.layers.Input(shape=(2,))
    xy_u = tf.keras.layers.Input(shape=(2,))
    xy_b = tf.keras.layers.Input(shape=(2,))
    xy_l = tf.keras.layers.Input(shape=(2,))
    xy_r = tf.keras.layers.Input(shape=(2,))
    
    ### initialize the differential operators
    dif_x = Dif_x(net_u)
    dif_y = Dif_y(net_v)
    
    ### obtain partial derivatives of u with respect to x and y
    U_x, U_y, U_xx, U_xy, U_yy = dif_x(xy)
    V_x, V_y, V_xx, V_xy, V_yy = dif_y(xy)
    U_u_x, U_u_y, U_u_xx, U_u_xy, U_u_yy = dif_x(xy_u)
    V_u_x, V_u_y, V_u_xx, V_u_xy, V_u_yy = dif_y(xy_u)
    U_b_x, U_b_y, U_b_xx, U_b_xy, U_b_yy = dif_x(xy_b)
    V_b_x, V_b_y, V_b_xx, V_b_xy, V_b_yy = dif_y(xy_b)
    U_l_x, U_l_y, U_l_xx, U_l_xy, U_l_yy = dif_x(xy_l)
    V_l_x, V_l_y, V_l_xx, V_l_xy, V_l_yy = dif_y(xy_l) 
    U_r_x, U_r_y, U_r_xx, U_r_xy, U_r_yy = dif_x(xy_r)
    V_r_x, V_r_y, V_r_xx, V_r_xy, V_r_yy = dif_y(xy_r) 
       
    ### Obtain the residuals from stress boundary conditions
    p = 'plain_stress'
    _, _, _, _, _, _, Gex, Gey = Material(U_x, U_y, V_x, V_y, U_xx, U_xy, U_yy, V_xx, V_xy, V_yy, E, mu, p)
    _, _, _, _, s_u_y, s_u_x, _, _ = Material(U_u_x, U_u_y, V_u_x, V_u_y, U_u_xx, U_u_xy, U_u_yy, V_u_xx, V_u_xy, V_u_yy, E, mu, p)
    _, _, _, _, s_b_y, s_b_x, _, _ = Material(U_b_x, U_b_y, V_b_x, V_b_y, U_b_xx, U_b_xy, U_b_yy, V_b_xx, V_b_xy, V_b_yy, E, mu, p)
    _, _, _, s_l_x, _, s_l_y, _, _ = Material(U_l_x, U_l_y, V_l_x, V_l_y, U_l_xx, U_l_xy, U_l_yy, V_l_xx, V_l_xy, V_l_yy, E, mu, p)
    _, _, _, s_r_x, _, s_r_y, _, _ = Material(U_r_x, U_r_y, V_r_x, V_r_y, U_r_xx, U_r_xy, U_r_yy, V_r_xx, V_r_xy, V_r_yy, E, mu, p)

    ### build up the PINN
    pinn = tf.keras.models.Model(inputs = [xy, xy_u, xy_b, xy_l, xy_r], \
            outputs = [Gex, Gey, s_u_x, s_u_y, s_b_x, s_b_y, s_l_x, s_l_y, s_r_x, s_r_y])

    return pinn