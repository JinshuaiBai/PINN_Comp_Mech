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

    time_start = time.time()
    hist, his_loss = l_bfgs_b.fit()
    time_end = time.time()
    
    T = time_end-time_start
    L = hist[1]
    it = hist[2]['funcalls']
    
    print('*************************************************\n')
    print('Time cost is', T, 's')
    print('Final loss is', L, '')
    print('Training converges by', it, 'iterations\n')
    print('*************************************************\n')
    
    return T, L, it, his_loss