import time

def Train(l_bfgs_b):
    """
    ====================================================================================================================

    Train function is to train the neural networks with the selected optimizer.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.

    [result]    [tuple]                 : The result returned by the optimiser;
    [his_loss]  [list]                  : History values of the loss terms;
    [t]         [float]                 : CPU time used for training;
    [l]         [float]                 : Final loss;
    [it]        [int]                   : The number of iterations for convergence.

    ====================================================================================================================
    """

    ### Execute the training process
    time_start = time.time()
    result, his_loss = l_bfgs_b.fit()
    time_end = time.time()

    ### Record the training time
    t = time_end-time_start

    ### Record the final loss
    l = result[1]

    ### Record the number of iterations for convergence
    it = result[2]['funcalls']

    print('\n*************************************************\n')
    print('Time cost is', t, 's')
    print('Final loss is', l, '')
    print('Training converges by', it, 'iterations\n')
    print('*************************************************\n')
    
    return t, l, it, his_loss