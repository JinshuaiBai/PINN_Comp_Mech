import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def Post_Process(net_u, pinn, his_loss):
    """
    ====================================================================================================================

    Post_Process function is to:
        1. Visualize the displacement, strain, and stress;
        2. Output results.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.
    
    [xy]        [array of float]        : Coordinates of all the sample points;
    [u]         [array of float]        : Displacement;
    [sigma]     [array of float]        : Stress;
    [epsilon]   [array of float]        : Strain;
    [net_u]     [keras model]           : The trained FNN for displacement u;
    [pinn]      [Keras model]           : The Physics-Informed Neural Network;
    [his_loss]  [list]                  : History values of the loss terms.

    ====================================================================================================================
    """

    plt.rcParams.update({'font.size': 9})
    
    xy = np.zeros((51, 1)).astype(np.float32)
    k = 0
    for i in range(0,51):
            xy[i, 0] = i * 1/50
    
    u = net_u.predict(xy) * xy
    temp = pinn.predict([xy for i in range(0,2)])
    sigma = temp[1]
    epsilon = temp[3]
    
    # plot figure for displacement u
    fig1 = plt.figure(1, figsize=(2,2), dpi = 600)
    plt.plot(np.array([0,1]),np.array([0,0.1]), color = '#0072BD',zorder = 1)
    plt.scatter(xy, u, s=10, c='#D95319',zorder = 2)
    plt.xlabel('$\it{x}$ (m)', fontdict = {'fontname': 'Calibri'})
    plt.ylabel('$\it{U}$ (m)', fontdict = {'fontname': 'Calibri'})
    # plt.title('Displacment', fontdict = {'fontname': 'Calibri'})
    plt.legend(['Analytic', 'PINN'])
    # plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0, 0.1])
    plt.savefig('u.tiff', dpi = 600, bbox_inches = 'tight')
    plt.show()
    
    # plot figure for strain epsilon
    fig2 = plt.figure(2, figsize=(2,2), dpi = 600)
    plt.plot(np.array([0,1]),np.array([0.1,0.1]), color = '#0072BD',zorder = 1)
    plt.scatter(xy, epsilon, s = 10, c = '#D95319',zorder = 2)
    plt.xlabel('$\it{x}$ (m)', fontdict = {'fontname': 'Calibri'})
    plt.ylabel('$\it{\epsilon}$', fontdict = {'fontname': 'Calibri'})
    # plt.title('Strain', fontdict = {'fontname': 'Calibri'})
    plt.legend(['Analytic', 'PINN'])
    # plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0.05, .15])
    plt.savefig('epsilon.tiff', dpi = 600, bbox_inches = 'tight')
    plt.show()
    
    # plot figure for stress sigma
    fig3 = plt.figure(3, figsize=(2,2), dpi = 600)
    plt.plot(np.array([0,1]),np.array([1,1]), color = '#0072BD',zorder = 1)
    plt.scatter(xy, sigma, s = 10, c = '#D95319',zorder = 2)
    plt.xlabel('$\it{x}$ (m)', fontdict = {'fontname': 'Calibri'})
    plt.ylabel('$\it{\sigma}$ $(N/m^2)$', fontdict = {'fontname': 'Calibri'})
    # plt.title('Stress', fontdict = {'fontname': 'Calibri'})
    plt.legend(['Analytic', 'PINN'])
    plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0.5, 1.5])
    plt.savefig('sigma.tiff', dpi = 600, bbox_inches = 'tight')
    plt.show()
    
    # plot figure for hist_loss
    fig4 = plt.figure(4, figsize=(8,3), dpi = 600)
    plt.plot(his_loss[0], color = 'r',zorder = 2,linestyle = '--')
    plt.plot(his_loss[1], color = 'b',zorder = 3,linestyle = '--')
    plt.plot(his_loss[0] + his_loss[1], color = 'k',zorder = 1)
    plt.yscale('log')
    plt.xlabel('Iteration', fontdict = {'fontname': 'Calibri'})
    plt.ylabel('Loss', fontdict = {'fontname': 'Calibri'})
    plt.title('Loss History', fontdict = {'fontname': 'Calibri'})
    plt.legend(['$L_{ge}$', '$L_{bc}$', '$L$'])
    plt.savefig('hist_loss.tiff', dpi = 600, bbox_inches = 'tight')
    plt.show()
    
    # output data in the 'out.mat' file
    scipy.io.savemat('out.mat', {'xy': xy, 'u': u, 'sigma': sigma, 'epsilon': epsilon})
    
    return None