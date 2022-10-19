import numpy as np
import math
import scipy.io

def Input_Info():
    """
    ====================================================================================================================

    This function is to load the problem information, including:
        1. Define the problem geometry;
        2. Define the material property;
        3. Define the boundary condition;
        4. Define the FNN settings.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [ns]        [int]                   : Total number of sample points;
    [dx]        [float]                 : Sample points interval;
    [x]         [Array of float32]      : Coordinates of all the sample points;
    [x1u]       [Array of float32]      : Coordinates of the sample points on the upper boundary of the plate;
    [x1b]       [Array of float32]      : Coordinates of the sample points on the bottom boundary of the plate;
    [x2u]       [Array of float32]      : Coordinates of the sample points on the left boundary of the plate;
    [x3u]       [Array of float32]      : Coordinates of the sample points on the right boundary of the plate;
    [x3b]       [Array of float32]      : Coordinates of the sample points on the right boundary of the plate;
    [x_train]   [List]                  : PINN input list, contains all the coordinates information;
    [s3u1]      [Array of float32]      : x direction traction boundary condition on the top boundary of the plate;
    [y_train]   [List]                  : PINN boundary condition list, contains the traction boundary condition;
    [n_input]   [int]                   : Number of inputs for the FNN;
    [n_output]  [int]                   : Number of outputs for the FNN;
    [layers]    [list]                  : Size of the FNN;
    [acti_fun]  [str]                   : The activation function used after each layer;
                                                        Available options:
                                                        'tanh'
                                                        'sigmoid'
                                                        'relu'
                                                        ... (more details in https://keras.io/api/layers/activations/)
    [k_init]    [str]                   : The kernel initialisation method.
    [NN_info]   [list]                  : Neural Network information list, contains the settings for the FNN;
    [E]         [float]                 : Young's module;
    [mu]        [float]                 : Poisson ratio.

    ====================================================================================================================
    """
    
    # sample points' interval
    dx = 1./20
    
    ### initialize sample points' coordinates
    C = scipy.io.loadmat('Coord.mat')
    x = C['xy']
    x1u = C['x1u']
    x1b = C['x1b']
    x2u = C['x2u']
    x2b = C['x2b']
    x3u = C['x3u']
    x3b = C['x3b']
    ns = C['n'][0, 0]
    
    ### Create the PINN input list
    x_train = [x, x1u, x1b, x2u, x2b, x3u, x3b]
    
    ### Define the material properties
    E = 1.
    mu = 0.25

    ### Define the traction boundary conditions
    s3u1 = np.multiply(np.cos(x3u[..., 0, np.newaxis] / 2 * math.pi), np.cos(x3u[..., 1, np.newaxis] / 2 * math.pi))
    
    ### Create the PINN boundary condition list
    y_train = [s3u1]
    
    ### Define the FNN settings
    n_input = 3
    n_output = 1
    layer = [np.array([20,20,20,20]), np.array([20,20,20,20]), np.array([20,20,20,20])]
    NN_info = [n_input, n_output, layer]
    
    print('*************************************************')
    print('Problem Info.')
    print('*************************************************\n')
    print(ns, 'sample points')
    print('The Young''s module is', E, '; The Possion''s ratio is', mu, '.\n')
    print('*************************************************')
    print('Neural Network Info.')
    print('*************************************************\n')
    print('net_u \nNumber of input:', n_input,', Number of output:', n_output, '.')
    print(len(layer[0]), 'hidden layers,', layer[0][0], ' neurons per layer.')
    print('net_v \nNumber of input:', n_input,', Number of output:', n_output, '.')
    print(len(layer[1]), 'hidden layers,', layer[0][1], ' neurons per layer.')
    print('net_w \nNumber of input:', n_input,', Number of output:', n_output, '.')
    print(len(layer[2]), 'hidden layers,', layer[0][2], ' neurons per layer.\n')
    print('*************************************************\n')
    return ns, x_train, y_train, E, mu, dx, NN_info