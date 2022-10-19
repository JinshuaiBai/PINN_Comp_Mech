import numpy as np
import math

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
    [xy]        [Array of float32]      : Coordinates of all the sample points;
    [xy_u]      [Array of float32]      : Coordinates of the sample points on the upper boundary of the plate;
    [xy_b]      [Array of float32]      : Coordinates of the sample points on the bottom boundary of the plate;
    [xy_l]      [Array of float32]      : Coordinates of the sample points on the left boundary of the plate;
    [xy_r]      [Array of float32]      : Coordinates of the sample points on the right boundary of the plate;
    [x_train]   [List]                  : PINN input list, contains all the coordinates information;
    [s_u_x]     [Array of float32]      : x direction traction boundary condition on the top boundary of the plate;
    [s_u_y]     [Array of float32]      : y direction traction boundary condition on the top boundary of the plate;
    [s_b_x]     [Array of float32]      : x direction traction boundary condition on the bottom boundary of the plate;
    [s_b_y]     [Array of float32]      : y direction traction boundary condition on the bottom boundary of the plate;
    [s_l_x]     [Array of float32]      : x direction traction boundary condition on the left boundary of the plate;
    [s_l_y]     [Array of float32]      : y direction traction boundary condition on the left boundary of the plate;
    [s_r_x]     [Array of float32]      : x direction traction boundary condition on the right boundary of the plate;
    [s_r_y]     [Array of float32]      : y direction traction boundary condition on the right boundary of the plate;
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
    
    ### Define the number of sample points
    ns_u = 51
    ns_l = 51
    ns = ns_u*ns_l
    
    ### Define the sample points' interval
    dx = 1./(ns_u-1)
    
    ### Initialize sample points' coordinates
    xy = np.zeros((ns, 2)).astype(np.float32)
    for i in range(0,ns_u):
        for j in range(0,ns_l):
            xy[i*ns_l+j, 0] = i * dx
            xy[i*ns_l+j, 1] = j * dx
    xy_u = np.hstack([np.linspace(0,1, ns_u).reshape(ns_u, 1).astype(np.float32), \
                      np.ones((ns_u,1)).astype(np.float32)])
    xy_b = np.hstack([np.linspace(0,1, ns_u).reshape(ns_u, 1).astype(np.float32), \
                      np.zeros((ns_u,1)).astype(np.float32)])
    xy_l = np.hstack([np.zeros((ns_l,1)).astype(np.float32), \
                  np.linspace(0,1, ns_l).reshape(ns_l, 1).astype(np.float32)])
    xy_r = np.hstack([np.ones((ns_l,1)).astype(np.float32), \
                  np.linspace(0,1, ns_l).reshape(ns_l, 1).astype(np.float32)])
    
    ### Create the PINN input list
    x_train = [ xy, xy_u, xy_b, xy_l, xy_r]
    
    ### Define the material properties
    E = 7.
    mu = 0.3
    
    ### Define the traction boundary conditions
    s_u_x = np.zeros((ns_u,1)).astype(np.float32)
    s_u_y = np.zeros((ns_u,1)).astype(np.float32)
    s_b_x = np.zeros((ns_u,1)).astype(np.float32)
    s_b_y = np.zeros((ns_u,1)).astype(np.float32)
    s_l_x = np.zeros((ns_l,1)).astype(np.float32)
    s_l_y = np.zeros((ns_l,1)).astype(np.float32)
    s_r_x = np.cos(xy_r[..., 1, np.newaxis]/2*math.pi)
    s_r_y = np.zeros((ns_l,1)).astype(np.float32)

    ### Create the PINN boundary condition list
    y_train = [ s_u_x, s_u_y, s_b_x, s_b_y, s_l_x, s_l_y, s_r_x, s_r_y ]
    
    ### Define the FNN settings
    n_input = 2
    n_output = 1
    layer = [np.array([ 20, 20, 20 ]), np.array([ 20, 20, 20 ])]
    NN_info = [n_input, n_output, layer]
    
    print('*************************************************')
    print('Problem Info.')
    print('*************************************************\n')
    print(ns, 'sample points')
    print('The Young''s module is', E,'; The Possion''s ratio is', mu,'.\n')
    print('*************************************************')
    print('Neural Network Info.')
    print('*************************************************\n')
    print('net_u \nNumber of input:',n_input,', Number of output:',n_output,'.')
    print(len(layer[0]),'hidden layers,',layer[0][0],' neurons per layer.')
    print('net_v \nNumber of input:',n_input,', Number of output:',n_output,'.')
    print(len(layer[1]),'hidden layers,',layer[0][1],' neurons per layer.\n')
    print('*************************************************\n')
    return ns, ns_u, ns_l, x_train, y_train, E, mu, dx, NN_info