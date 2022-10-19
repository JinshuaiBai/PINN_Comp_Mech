import tensorflow as tf
from lib.Pre.Material import Material
from lib.Pre.Dif_op import Dif

def PINN(net_u, E):
    """
    ====================================================================================================================

    This function is to initialize a PINN.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [net_u]     [keras model]           : The trained FNN for displacement u;
    [xy]        [Keras input layer]     : Coordinates of all the sample points;
    [xy_r]      [Keras input layer]     : Coordinate of the sample point at the right tip of the rod;
    [U_r]       [Keras tensor]          : Displacement at the right tip of the rod;
    [U_x]       [Keras tensor]          : First-order derivative of displacement with respect to x direction;
    [U_xx]      [Keras tensor]          : Second-order derivative of displacement with respect to x direction;
    [U_r_x]     [Keras tensor]          : First-order derivative of displacement at the right tip of the rod with
                                          respect to x direction;
    [U_r_xx]    [Keras tensor]          : Second-order derivative of displacement at the right tip of the rod with
                                          respect to x direction;
    [epsilon]   [Keras tensor]          : Strain;
    [sigma]     [Keras tensor]          : Stress;
    [Ge]        [Keras tensor]          : Residual from the equilibrium equation;
    [sigma_r]   [Keras tensor]          : Stress at the right tip of the rod;
    [E]         [float]                 : Young's module.

    ====================================================================================================================
    """

    ### declare PINN's inputs
    xy = tf.keras.layers.Input(shape=(1,))
    xy_r = tf.keras.layers.Input(shape=(1,))
    
    ### initialize the differential operators
    Dif_u = Dif(net_u)
    
    ### obtain the displacment at the right tip of the rod
    u_r = net_u(xy_r) * xy_r
    
    ### obtain partial derivatives of u with respect to x
    u_x, u_xx = Dif_u(xy)
    u_r_x, u_r_xx = Dif_u(xy_r)
       
    ### obtain the residuals from the governing equation and traction boundary condition
    epsilon, sigma, Ge = Material(u_x, u_xx, E)
    _, sigma_r, _ = Material(u_r_x, u_r_xx, E)
    
    ### build up the PINN
    pinn = tf.keras.models.Model(inputs=[xy, xy_r], outputs=[Ge, sigma_r, sigma, epsilon, u_r])
        
    return pinn