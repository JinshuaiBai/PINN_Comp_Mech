import tensorflow as tf

def FNN(n_input, n_output, layers, acti_fun='tanh', k_init='LecunNormal'):
    """
    ====================================================================================================================

    This function is to initialise a FNN.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [n_input]   [int]                   : Number of inputs for the FNN;
    [n_output]  [int]                   : Number of outputs for the FNN;
    [layers]    [list]                  : Size of the FNN;
    [acti_fun]  [str]                   : The activation function used after each layer;
                                                        Available options:
                                                        'tanh'
                                                        'sigmoid'
                                                        'relu'
                                                        ... (more details in https://keras.io/api/layers/activations/)
    [k_init]    [str]                   : The kernel initialisation method;
    [x]         [Keras layer]           : Input of the FNN;
    [temp]      [Keras layer]           : Hidden layers of the FNN;
    [y]         [Keras layer]           : Output of the FNN;
    [net]       [Keras model]           : The built FNN.

    ====================================================================================================================
    """

    ### Setup the input layer of the FNN
    x = tf.keras.layers.Input(shape=(n_input))
    
    ### Setup the hidden layers of the FNN
    temp = x
    for l in layers:
        temp = tf.keras.layers.Dense(l, activation = acti_fun, kernel_initializer=k_init)(temp)
    
    ### Setup the output layers of the FNN
    y = tf.keras.layers.Dense(n_output, kernel_initializer=k_init)(temp)

    ### Combine the input, hidden, and output layers to build up a FNN
    net = tf.keras.models.Model(inputs=x, outputs=y)

    return net
