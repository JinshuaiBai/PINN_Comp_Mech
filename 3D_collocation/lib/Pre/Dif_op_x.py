import tensorflow as tf

class Dif_x(tf.keras.layers.Layer):
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

        This function is to initialise for differential operator.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [fnn]       [Keras model]           : The Feedforward Neural Network.

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
        [U]         [Keras tensor]          : The displacement predictions;
        [U_x]       [Keras tensor]          : The first-order derivative of the u with respect to the x;
        [U_y]       [Keras tensor]          : The first-order derivative of the u with respect to the y;
        [U_z]       [Keras tensor]          : The first-order derivative of the u with respect to the z;
        [U_xx]      [Keras tensor]          : The second-order derivative of the u with respect to the x;
        [U_xy]      [Keras tensor]          : The second-order derivative of the u with respect to the xy;
        [U_xz]      [Keras tensor]          : The second-order derivative of the u with respect to the xz;
        [U_yy]      [Keras tensor]          : The second-order derivative of the u with respect to the y;
        [U_yz]      [Keras tensor]          : The second-order derivative of the u with respect to the yz;
        [U_zz]      [Keras tensor]          : The second-order derivative of the u with respect to the z.

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
                U = temp * x

            ### Obtain the first-order derivative of the output with respect to the input
            U_x = g.gradient(U, x)
            U_y = g.gradient(U, y)
            U_z = g.gradient(U, z)
            del g

        ### Obtain the second-order derivative of the output with respect to the input
        U_xx = gg.gradient(U_x, x)
        U_xy = gg.gradient(U_x, y)
        U_xz = gg.gradient(U_x, z)
        U_yy = gg.gradient(U_y, y)
        U_yz = gg.gradient(U_y, z)
        U_zz = gg.gradient(U_z, z)
        del gg

        return U_x, U_y, U_z, U_xx, U_xy, U_xz, U_yy, U_yz, U_zz