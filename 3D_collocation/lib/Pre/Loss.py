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
    [l_x]       [Keras tensor]          : The loss term from the traction boundary on x surfaces;
    [l_y]       [Keras tensor]          : The loss term from the traction boundary on y surfaces;
    [l_z]       [Keras tensor]          : The loss term from the traction boundary on z surfaces;
    [l2]        [Keras tensor]          : The loss term from the overall traction boundaries;
    [loss]      [Keras tensor]          : The final loss.

    ====================================================================================================================
    """

    ### Residual from the governing equation
    l1 = (tf.reduce_sum(tf.square(y_p[0])) +
          tf.reduce_sum(tf.square(y_p[1])) +
          tf.reduce_sum(tf.square(y_p[2])))

    ### Residual from the traction boundary condition on x surfaces
    l_x = (tf.reduce_sum(tf.square(y_p[3])) +
          tf.reduce_sum(tf.square(y_p[4])) +
          tf.reduce_sum(tf.square(y_p[5])) +
          tf.reduce_sum(tf.square(y_p[6])) +
          tf.reduce_sum(tf.square(y_p[7])))

    ### Residual from the traction boundary condition on y surfaces
    l_y = (tf.reduce_sum(tf.square(y_p[8])) +
          tf.reduce_sum(tf.square(y_p[9])) +
          tf.reduce_sum(tf.square(y_p[10])) +
          tf.reduce_sum(tf.square(y_p[11])) +
          tf.reduce_sum(tf.square(y_p[12])))

    ### Residual from the traction boundary condition on z surfaces
    l_z = (tf.reduce_sum(tf.square(y_p[13] - y[0])) +
          tf.reduce_sum(tf.square(y_p[14])) +
          tf.reduce_sum(tf.square(y_p[15])) +
          tf.reduce_sum(tf.square(y_p[16])) +
          tf.reduce_sum(tf.square(y_p[17])))

    ### Overall residual from the traction boundaries
    l2 = l_x + l_y + l_z

    ### Final loss
    loss = l1 + l2

    return loss, l1, l2
