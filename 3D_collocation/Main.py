import tensorflow as tf
import os
from lib.Pre_Process import Pre_Process
from lib.Train import Train
from lib.Post_Process import Post_Process
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#%%
"""
=================================================================================================================================
    This code is for the 3D stretching cube problem in "An introduction to programming physics-informed
    neural network-based computational solid mechanics". 
    DOI: https://doi.org/10.48550/arXiv.2210.09060
    
    A 3D stretching cube problem is modeled here. The length of the cube is L = 2 m. The in-plain
    distribute force, F(x,y) = cos(pi*x/2)cos(pi*y/2) N/m^2, is applied on the top surface of the
    cube. 9261 sample points are uniformly generated in the computational domain with 0.02 m spacing
    in all directions. We use three FNNs with the same structures to respectively predict the 
    displacement field u, v and w. The Young's module is E = 10 Pa and the Poisson ratio is 0.25.
    
    The code includes three parts:
        1. Pre-Processing part  : Initialize the geometry, material properties, boundary conditions,
                                  feedforward neural networks, physics-informed neural networks 
                                  and optimizer.
        2. Training part        : Train the neural networks with the selected optimizer.
        3. Post-Processing part : Visualize and output the results.

    Libraries used in this code are as follow:
        Name             Source                             Location
        'TensorFlow'     https://www.tensorflow.org/
        'NumPy'          https://numpy.org/
        'SciPy'          https://scipy.org/
        'Matplotlib'     https://matplotlib.org/
        'time'           In Python3
        'os'             In Python3
        'Pre_Process'    Self developed                     ./lib
        'Train'          Self developed                     ./lib
        'Post_Process'   Self developed                     ./lib
        'Input_Info'     Self developed                     ./lib/Pre/
        'FNN'            Self developed                     ./lib/Pre/
        'PINN'           Self developed                     ./lib/Pre/
        'L_BFGS_B'       Self developed                     ./lib/Pre/
        'Loss'           Self developed                     ./lib/Pre/
        
        
    This code is developed by @Jinshuai Bai and @Yuantong Gu. For more details, please contact: 
    jinshuai.bai@hdr.qut.edu.au
    yuantong.gu@qut.edu.au
=================================================================================================================================  
"""
if __name__ == '__main__':
    """
        Pre_Process() function is to:
            1. Define the geometry of the problem
            2. Define the material properties
            3. Define the boundary conditions
            4. Initialize the neural networks
            5. Initialize the optimier
    """
    
    net_u, net_v, net_w, pinn, l_bfgs_b = Pre_Process()
    
    """
        Train() function is to train the PINN with the selected optimizer
    """
    
    T, L, it, his_loss = Train(l_bfgs_b)
    
    """
        Post_Process() function is to:
            1. Visualize the predicted field variables
            2. Output results
    """
    
    Post_Process(net_u, net_v, net_w, pinn, his_loss)
