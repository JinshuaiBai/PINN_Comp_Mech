import numpy as np

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
    [xy_r]      [Array of float32]      : Coordinates of the sample points on the right tip of the rod;
    [x_train]   [List]                  : PINN input list, contains all the coordinates information;
    [s_r_x]     [Array of float32]      : Traction boundary condition on the right tip of the rod;
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
    [E]         [float]                 : Young's module.
        
    ====================================================================================================================
    """
    
    ### Define the number of sample points
    ns = 51
    
    ### Define the sample points' interval
    dx = 1./(ns-1)
    
    ### Initialize sample points' coordinates
    xy = np.zeros((ns, 1)).astype(np.float32)
    for i in range(0, ns):
        xy[i, 0] = i * dx
    xy_r = np.array([1.])
    
    ### Create the PINN input list
    x_train = [xy, xy_r]
    
    ### Define the Young's modulus
    E = 10.
    
    ### Define the traction boundary condition at the right tip of the rod
    s_r_x = 1.
    
    ### Create the PINN boundary condition list
    y_train = [s_r_x]
    
    ### Define the FNN settings
    n_input = 1
    n_output = 1
    layer = [np.array([5, 5, 5])]
    acti_fun = 'tanh'
    k_init = 'LecunNormal'
    NN_info = [n_input, n_output, layer, acti_fun, k_init]

    ### Visualise the summary of the problem setup
    print('*************************************************')
    print('Problem Info.')
    print('*************************************************\n')
    print(ns, 'sample points')
    print('The Young''s module is', E,'.\n')
    print('*************************************************')
    print('Neural Network Info.')
    print('*************************************************\n')
    print('net_u \nNumber of input:',n_input,', Number of output:',n_output,'.')
    print(len(layer[0]),'hidden layers,',layer[0][0],' neurons per layer.\n')
    print('*************************************************\n')

    return ns, x_train, y_train, E, dx, NN_info