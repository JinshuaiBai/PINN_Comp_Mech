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
    [u_r]       [Keras tensor]          : Displacement at the right tip of the rod;
    [e1]        [Keras tensor]          : Normal strain for x direction;
    [e2]        [Keras tensor]          : Normal strain for y direction;
    [e12]       [Keras tensor]          : Shear strain;
    [s1]        [Keras tensor]          : Normal stress for x direction;
    [s2]        [Keras tensor]          : Normal stress for y direction;
    [s12]       [Keras tensor]          : Shear stress;
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

    ###obtain the displacment at the right tip of the rod
    u_r = net_u(xy_r)*xy_r[...,0,tf.newaxis]
    
    ### obtain partial derivatives of u with respect to x and y
    U_x, U_y, U_xx, U_xy, U_yy = dif_x(xy)
    V_x, V_y, V_xx, V_xy, V_yy = dif_y(xy)
       
    ### Obtain the residuals from the governing equation and stress boundary conditions
    p = 'plain_stress'
    e1, e2, e12, s1, s2, s12, _, _ = Material(U_x, U_y, V_x, V_y, U_xx, U_xy, U_yy, V_xx, V_xy, V_yy, E, mu, p)

    ### build up the PINN
    pinn = tf.keras.models.Model(inputs = [xy, xy_u, xy_b, xy_l, xy_r], \
            outputs = [e1, e2, e12, s1, s2, s12, u_r])

    return pinn