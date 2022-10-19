import tensorflow as tf

class Dif(tf.keras.layers.Layer):
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

    def call(self, x):
        """
        ================================================================================================================

        This function is to calculate the differential terms.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [x]         [Keras model]           : The coordinate array;
        [temp]      [Keras tensor]          : The intermediate output from the FNN;
        [u]         [Keras tensor]          : The displacement predictions;
        [u_x]       [Keras tensor]          : The first-order derivative of the u with respect to the x;
        [u_xx]      [Keras tensor]          : The second-order derivative of the u with respect to the x.

        ================================================================================================================
        """

        ### Apply the GradientTape function
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)

                ### Obtain the intermediate output from the FNN
                temp = self.fnn(x)

                ### Calculate the displacement output by times the coordinate to naturally satisfy the displacement
                ### boundary condition
                u = temp * x

            ### Obtain the first-order derivative of the output with respect to the input
            u_x = g.gradient(u, x)
            del g

        ### Obtain the second-order derivative of the output with respect to the input
        u_xx = gg.gradient(u_x, x)
        del gg

        return u_x, u_xx
