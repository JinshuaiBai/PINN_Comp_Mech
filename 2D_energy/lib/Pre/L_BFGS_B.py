import scipy.optimize
import numpy as np
import tensorflow as tf
from lib.Pre.Loss import Energy_Loss
class L_BFGS_B:
    """
        ====================================================================================================================

        This is the class for the L-BFGS-B optimiser. We adopt core algorithm of the L-BFGS-B algorithm is provided by the
        Scipy library. This class include 5 functions, including:
            1. __init__()         : Initialise the parameters for the L-BFGS-B optimiser;
            2. pi_loss()          : Calculate the physics-informed loss;
            3. loss_grad()        : Obtain the gradients of the physics-informed loss with respect to the weighs and biases;
            4. set_weights()      : Set the modified weights and biases back to the neural network structure;
            5. fit()              : Execute training process.

        ====================================================================================================================
    """

    def __init__(self, pinn, x_train, y_train, dx, factr=10, pgtol=1e-10, m=50, maxls=50, maxfun=40000):
        """
        ================================================================================================================

        This function is to initialise the parameters used in the L-BFGS-B optimiser.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [pinn]      [Keras model]           : The Physics-informed neural network;
        [x_train]   [list]                  : PINN input list, contains all the coordinates information;
        [y_train]   [list]                  : PINN boundary condition list, contains the traction boundary condition;
        [dx]        [float]                 : Sample points interval;
        [factr]     [int]                   : The optimiser option. Please refer to SciPy;
        [pgtol]     [float]                 : The optimiser option. Please refer to SciPy;
        [m]         [int]                   : The optimiser option. Please refer to SciPy;
        [maxls]     [int]                   : The optimiser option. Please refer to SciPy;
        [maxfun]    [int]                   : Maximum number of iterations for training;
        [iter]      [int]                   : Number of training iterations;
        [his_l1]    [list of float32]       : History values of the l1 loss term;
        [his_l2]    [list of float32]       : History values of the l1 loss term.

        ================================================================================================================
        """

        ### Initialise the parameters
        self.pinn = pinn
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.dx = dx
        self.factr = factr
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.maxfun = maxfun
        self.metrics = ['loss']
        self.iter = 0
        self.his_loss_ge = []
        self.his_loss_bc = []

    def pi_loss(self, weights):
        """
        ================================================================================================================

        This function is to calculate the physics-informed loss.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [weights]   [list]                  : The weights and biases;
        [pinn]      [Keras tensor]          : The Physics-informed neural network;
        [x_train]   [list]                  : PINN input list, contains all the coordinates information;
        [y_train]   [list]                  : PINN boundary condition list, contains the traction boundary condition;
        [loss]      [Keras tensor]          : Current value of the physics-informed loss;
        [l1]        [Keras tensor]          : The l1 loss term;
        [l2]        [Keras tensor]          : The l2 loss term;
        [grads]     [Keras tensor]          : The gradients of the physics-informed loss with respect to weights and
                                              biases;
        [iter]      [int]                   : Number of training iterations.

        ================================================================================================================
        """

        ### Update the weights and biases to the FNN
        self.set_weights(weights)

        ### Calculate the physics-informed loss and its gradients with respect to weights and biases
        loss, grads, l1, l2 = self.loss_grad(self.x_train, self.y_train)

        ### Count number of the training iteration
        self.iter = self.iter + 1.

        ### Print the loss terms every 10 training iterations
        if self.iter % 10 == 0:
            print('Iter: %d   L1 = %.4g   L2 = %.4g' % (self.iter, l1.numpy(), l2.numpy()))

        ### Convert loss and grads from Keras tensor to ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype('float64')

        ### Save the current loss term in different np.array
        self.his_loss_ge.append(l1)
        self.his_loss_bc.append(l2)

        return loss, grads

    @tf.function
    def loss_grad(self, x, y):
        """
        ================================================================================================================

        This function is to obtain the gradients of the physics-informed loss with respect to the weighs and biases.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [pinn]      [Keras model]           : The Physics-informed neural network;
        [x_train]   [list]                  : PINN input list, contains all the coordinates information;
        [y_train]   [list]                  : PINN boundary condition list, contains the traction boundary condition;
        [y_p]       [list]                  : List of predictions from the PINN;
        [loss]      [Keras tensor]          : Current value of the physics-informed loss;
        [l1]        [Keras tensor]          : The l1 loss term;
        [l2]        [Keras tensor]          : The l2 loss term;
        [dx]        [float]                 : Sample points interval;
        [grads]     [Keras tensor]          : The gradients of the physics-informed loss with respect to weights and
                                              biases.

        ================================================================================================================
        """

        with tf.GradientTape() as g:

            ### Predict outputs from the current PINN
            y_p = self.pinn(x)

            ### Apply the Energy Loss function
            loss, l1, l2 = Energy_Loss(y_p,y,self.dx)

        ### Obtain the gradients through automatic differentiation
        ### (GradientTape function provided by the TensorFlow)
        grads = g.gradient(loss, self.pinn.trainable_variables)

        return loss, grads, l1, l2

    def set_weights(self, flat_weights):
        """
        ================================================================================================================

        This function is to Set the modified weights and biases back to the neural network structure.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [pinn]      [keras model]           : The Physics-informed neural network;
        [shapes]    [list]                  : The shapes of neural network's weights and biases;
        [weights]   [list]                  : The weights and biases.

        ================================================================================================================
        """

        ### Obtain the shapes of neural network's weights and biases
        shapes = [w.shape for w in self.pinn.get_weights()]

        ### Compute splitting indices
        split_ids = np.cumsum([np.prod(shape) for shape in [0] + shapes])

        ### Reshape the modified weights and biases to fit the neural network structure
        weights = [flat_weights[from_id:to_id].reshape(shape)
                   for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes)]

        ### Set weights and biases to the neural network
        self.pinn.set_weights(weights)

        return None

    def fit(self):
        """
        ================================================================================================================

        This function is to execute training process.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [pinn]      [Keras model]           : The Physics-informed neural network;
        [ini_w]     [ndarray]               : The initial weights and biases;
        [pi_loss]   [function]              : The function that calculates the physics-informed loss (line 61);
        [factr]     [int]                   : The optimiser option. Please refer to SciPy;
        [pgtol]     [float]                 : The optimiser option. Please refer to SciPy;
        [m]         [int]                   : The optimiser option. Please refer to SciPy;
        [maxls]     [int]                   : The optimiser option. Please refer to SciPy;
        [maxfun]    [int]                   : Maximum number of iterations for training;
        [result]    [tuple]                 : The result returned by the optimiser;
        [his_l1]    [list of float32]       : History values of the l1 loss term;
        [his_l2]    [list of float32]       : History values of the l2 loss term.

        ================================================================================================================
        """

        ### Get initial weights and biases
        initial_weights = np.concatenate([ w.flatten() for w in self.pinn.get_weights() ])

        ### Optimise the weights and biases via the L-BFGS-B optimiser
        print('Optimizer: L-BFGS-B (Provided by Scipy package)')
        print('Initializing ...')
        result = scipy.optimize.fmin_l_bfgs_b(func=self.pi_loss, x0=initial_weights,
            factr=self.factr, pgtol=self.pgtol, m=self.m, maxls=self.maxls, maxfun=self.maxfun)

        return result, [np.array(self.his_loss_ge), np.array(self.his_loss_bc)]