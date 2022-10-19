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
    l1 = (tf.reduce_mean(tf.square(y_p[0])) + \
        tf.reduce_mean(tf.square(y_p[1])))

    ### Residual from the traction boundary condition
    l2 = ((tf.reduce_mean(tf.square(y_p[2]-y[0])) + \
        tf.reduce_mean(tf.square(y_p[3]-y[1]))) + \
        tf.reduce_mean(tf.square(y_p[4]-y[2])) + \
        tf.reduce_mean(tf.square(y_p[7]-y[5])) + \
        (tf.reduce_mean(tf.square(y_p[8]-y[6])) + \
        tf.reduce_mean(tf.square(y_p[9]-y[7]))))

    ### Final loss
    loss = l1 + l2

    return loss, l1, l2