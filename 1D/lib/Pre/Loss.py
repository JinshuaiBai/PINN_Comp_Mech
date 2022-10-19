import tensorflow as tf

def Collocation_Loss(y_p, y):
    """
    ====================================================================================================================

    Collocation loss function

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [y_p]       [list]                  : Outputs from the PINN;
    [y]         [list]                  : The ground truth data;
    [l1]        [Keras tensor]          : The loss term from the equilibrium equation;
    [l2]        [Keras tensor]          : The loss term from the traction boundary condition;
    [loss]      [Keras tensor]          : The final loss

    ====================================================================================================================
    """

    ### Residual from the governing equation
    l1 = tf.reduce_mean(tf.square(y_p[0]))

    ### Residual from the traction boundary condition
    l2 = tf.reduce_mean(tf.square(y_p[1]-y[0]))

    ### Final loss
    loss = l1 + l2

    return loss, l1, l2
    
def Energy_Loss(y_p, y, dx):
    """
    ====================================================================================================================

    Energy-based loss function

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [y_p]       [list]                  : Outputs from the PINN;
    [y]         [list]                  : The ground truth data;
    [dx]        [float]                 : Sample points interval;
    [l1]        [Keras tensor]          : The internal potential energy;
    [l2]        [Keras tensor]          : The potential energy of the external traction force;
    [loss]      [Keras tensor]          : The final loss

    ====================================================================================================================
    """

    ### Internal potential energy
    l1 = 0.5 * dx * tf.reduce_sum(y_p[2] * y_p[3])
    ### Potential energy of the external force
    l2 = tf.reduce_sum(y_p[4] * y[0])
    ### Final Loss
    loss = l1 - l2

    return loss, l1, l2
