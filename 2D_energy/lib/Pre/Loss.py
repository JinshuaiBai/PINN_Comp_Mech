import tensorflow as tf

def Energy_Loss(y_p, y, dx):
    """
      ====================================================================================================================

      Energy-based loss function

      --------------------------------------------------------------------------------------------------------------------

      Name        Type                    Info.

      [y_p]       [list]                  : Outputs from the PINN;
      [y]         [list]                  : The ground truth data;
      [dx]        [float]                 : Sample points interval for x-axis;
      [dy]        [float                  : Sample points interval for y-axis;
      [l1]        [Keras tensor]          : The internal potential energy;
      [l2]        [Keras tensor]          : The potential energy of the external traction force;
      [loss]      [Keras tensor]          : The final loss

      ====================================================================================================================
      """
    ### Internal potential energy
    l1 = 0.5 * dx * dx * tf.reduce_sum((y_p[0] * y_p[3])+(y_p[1] * y_p[4])+(y_p[2] * y_p[5]))
    ### Potential energy of the external force
    l2 = tf.reduce_sum(y_p[6] * y[6]) * dx

    ### Final loss
    loss = l1 - l2

    return loss, l1, l2