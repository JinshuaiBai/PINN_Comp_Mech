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
        [V_xx]      [Keras tensor]          : The second-order derivative of the v with respect to the x;
        [V_xy]      [Keras tensor]          : The second-order derivative of the v with respect to the xy;
        [V_yy]      [Keras tensor]          : The second-order derivative of the v with respect to the y.

        ================================================================================================================
        """

        ### Divide the coordinate array into x and y components
        x, y = (xy[..., i, tf.newaxis] for i in range(xy.shape[-1]))

        ### Apply the GradientTape function
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            gg.watch(y)
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(y)

                ### Obtain the intermediate output from the FNN
                temp = self.fnn(tf.concat([x, y], axis=-1))

                ### Calculate the displacement output by times the coordinate to naturally satisfy the displacement
                ### boundary condition
                V = temp * y

            ### Obtain the first-order derivative of the output with respect to the input
            V_x = g.gradient(V, x)
            V_y = g.gradient(V, y)
            del g

        ### Obtain the second-order derivative of the output with respect to the input
        V_xx = gg.gradient(V_x, x)
        V_xy = gg.gradient(V_x, y)
        V_yy = gg.gradient(V_y, y)
        del gg

        return V_x, V_y, V_xx, V_xy, V_yy
