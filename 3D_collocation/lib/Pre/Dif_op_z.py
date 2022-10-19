import tensorflow as tf

class Dif_z(tf.keras.layers.Layer):
    """
    ====================================================================================================================

    This is the class for calculating the differential terms of the FNN's output with respect to the FNN's input. We
    adopt the GradientTape function provided by the TensorFlow library to do the automatic differentiation.
    This class include 2 functions, including:
        1. __init__()         : Initialise the parameters for differential operator;
        2. call()             : Calculate the differential terms.

    ====================================================================================================================
    """

    def __init__(self, fnn, **kwargs):
        """
        ================================================================================================================

        Options:
            Name        Type                    Size        Info.

            'fnn'       [keras model]           \           : The Feedforward Neural Network.

        ================================================================================================================
        """
        self.fnn = fnn
        super().__init__(**kwargs)

    @tf.function
    def call(self, xy):
        """
        ================================================================================================================

        This function is to calculate the differential terms.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [xy]        [Keras model]           : The coordinate array;
        [temp]      [Keras tensor]          : The intermediate output from the FNN;
        [W]         [Keras tensor]          : The displacement predictions;
        [W_x]       [Keras tensor]          : The first-order derivative of the w with respect to the x;
        [W_y]       [Keras tensor]          : The first-order derivative of the w with respect to the y;
        [W_z]       [Keras tensor]          : The first-order derivative of the w with respect to the z;
        [W_xx]      [Keras tensor]          : The second-order derivative of the w with respect to the x;
        [W_xy]      [Keras tensor]          : The second-order derivative of the w with respect to the xy;
        [W_xz]      [Keras tensor]          : The second-order derivative of the w with respect to the xz;
        [W_yy]      [Keras tensor]          : The second-order derivative of the w with respect to the y;
        [W_yz]      [Keras tensor]          : The second-order derivative of the w with respect to the yz;
        [W_zz]      [Keras tensor]          : The second-order derivative of the w with respect to the z.

        ================================================================================================================
        """

        ### Divide the coordinate array into x, y and z components
        x, y, z = (xy[..., i, tf.newaxis] for i in range(xy.shape[-1]))

        ### Apply the GradientTape function
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            gg.watch(y)
            gg.watch(z)
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(y)
                g.watch(z)

                ### Obtain the intermediate output from the FNN
                temp = self.fnn(tf.concat([x, y, z], axis=-1))

                ### Calculate the displacement output by times the coordinate to naturally satisfy the displacement
                ### boundary condition
                V = temp * z

            ### Obtain the first-order derivative of the output with respect to the input
            W_x = g.gradient(V, x)
            W_y = g.gradient(V, y)
            W_z = g.gradient(V, z)
            del g

        ### Obtain the second-order derivative of the output with respect to the input
        W_xx = gg.gradient(W_x, x)
        W_xy = gg.gradient(W_x, y)
        W_xz = gg.gradient(W_x, z)
        W_yy = gg.gradient(W_y, y)
        W_yz = gg.gradient(W_y, z)
        W_zz = gg.gradient(W_z, z)
        del gg

        return W_x, W_y, W_z, W_xx, W_xy, W_xz, W_yy, W_yz, W_zz