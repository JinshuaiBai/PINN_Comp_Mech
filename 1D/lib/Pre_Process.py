from lib.Pre.Input_Info import Input_Info
from lib.Pre.FNN import FNN
from lib.Pre.PINN import PINN
from lib.Pre.L_BFGS_B import L_BFGS_B

def Pre_Process():
    """
    ====================================================================================================================

    Pre_Process function is to:
        1. Load the problem information;
        2. Build up the FNN;
        3. Build up the PINN;
        4. Initialize the L-BFGS-B optimiser.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [ns]        [int]                   : Total number of sample points;
    [ns_u]      [int]                   : Number of sample points on top boundary of the beam;
    [ns_l]      [int]                   : Number of sample points on left boundary of the beam;
    [dx]        [float]                 : Sample points interval;
    [x_train]   [List]                  : PINN input list, contains all the coordinates information;
    [y_train]   [List]                  : PINN boundary condition list, contains all the traction boundary conditions;
    [NN_info]   [list]                  : Neural Network information list, contains the settings for the FNN;
    [E]         [float]                 : Young's module;
    [net_u]     [Keras model]           : The built FNN;
    [pinn]      [Keras model]           : The built PINN;
    [l_bfgs_b]  [class]                 : The initialised L-BFGS-B optimiser.
        
    ====================================================================================================================
    """
    
    ### Input information
    ns, x_train, y_train, E, dx, NN_info = Input_Info()
    
    ### Initialize the Feedforward Neural Networks
    net_u = FNN(n_input=NN_info[0], n_output=NN_info[1], layers=NN_info[2][0], acti_fun=NN_info[3], k_init=NN_info[4])
    
    ### Initialize the Physics-informed Neural Network
    pinn = PINN(net_u, E)
    
    ### Initialize the L-BFGS-B optimizer
    l_bfgs_b = L_BFGS_B(pinn, x_train, y_train, dx)

    return net_u, pinn, l_bfgs_b