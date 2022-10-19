import tensorflow as tf

class Dif_y(tf.keras.layers.Layer):
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
        [V]         [Keras tensor]          : The displacement predictions;
        [V_x]       [Keras tensor]          : The first-order derivative of the v with respect to the x;
        [V_y]       [Keras tensor]          : The first-order derivative of the v with respect to the y;
        [V_z]       [Keras tensor]          : The first-order derivative of the v with respect to the z;
        [V_xx]      [Keras tensor]          : The second-order derivative of the v with respect to the x;
        [V_xy]      [Keras tensor]          : The second-order derivative of the v with respect to the xy;
        [V_xz]      [Keras tensor]          : The second-order derivative of the v with respect to the xz;
        [V_yy]      [Keras tensor]          : The second-order derivative of the v with respect to the y;
        [V_yz]      [Keras tensor]          : The second-order derivative of the v with respect to the yz;
        [V_zz]      [Keras tensor]          : The second-order derivative of the v with respect to the z.

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
                V = temp * y

            ### Obtain the first-order derivative of the output with respect to the input
            V_x = g.gradient(V, x)
            V_y = g.gradient(V, y)
            V_z = g.gradient(V, z)
            del g

        ### Obtain the second-order derivative of the output with respect to the input
        V_xx = gg.gradient(V_x, x)
        V_xy = gg.gradient(V_x, y)
        V_xz = gg.gradient(V_x, z)
        V_yy = gg.gradient(V_y, y)
        V_yz = gg.gradient(V_y, z)
        V_zz = gg.gradient(V_z, z)
        del gg

        return V_x, V_y, V_z, V_xx, V_xy, V_xz, V_yy, V_yz, V_zz