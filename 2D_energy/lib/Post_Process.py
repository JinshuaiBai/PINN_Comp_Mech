import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def Post_Process(net_u, net_v, pinn, his_loss):
    """
    ====================================================================================================================

    Post_Process function is to:
        1. Visualize the displacement, strain, and stress;
        2. Output results.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.
    
    [xy]        [array of float]        : Coordinates of all the sample points;
    [u]         [array of float]        : Displacement in x direction;
    [v]         [array of float]        : Displacement in y direction;
    [s11]       [array of float]        : Normal Stress in x direction;
    [s22]       [array of float]        : Normal Stress in y direction;
    [s12]       [array of float]        : Shear Stress on xy plane;
    [net_u]     [keras model]           : The trained FNN for displacement u;
    [net_v]     [keras model]           : The trained FNN for displacement v;
    [pinn]      [Keras model]           : The Physics-Informed Neural Network;
    [his_loss]  [list]                  : History values of the loss terms.

    ====================================================================================================================
    """

    xy = np.zeros((201*201, 2)).astype(np.float32)
    k = 0
    for i in range(0,201):
        for j in range(0,201):
                xy[k, 0] = i * 1/200
                xy[k, 1] = j * 1/200
                k = k+1
    
    u = net_u.predict(xy) * xy[..., 0, np.newaxis]
    v = net_v.predict(xy) * xy[..., 1, np.newaxis]
    temp = pinn.predict([ xy for i in range(0,5) ])
    s11 = temp[3]
    s22 = temp[4]
    s12 = temp[5]

    ### plot figure for displacement u
    fig1 = plt.figure(1)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = u, cmap = 'jet', vmin = 0, vmax = 0.15)
    plt.axis('equal')
    plt.colorbar()
    plt.title('u')
    plt.savefig('u.tiff', dpi = 600)
    
    ### plot figure for displacement v
    fig2 = plt.figure(2)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = v, cmap = 'jet', vmin = -4e-2, vmax = 0)
    plt.axis('equal')
    plt.colorbar()
    plt.title('v')
    plt.savefig('v.tiff', dpi = 600)
    
    ### plot figure for stress sigma_x
    fig3 = plt.figure(3)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = s11, cmap = 'jet', vmin = 0, vmax = 1)
    plt.axis('equal')
    plt.colorbar()
    plt.title(r'$\sigma_{x}$')
    plt.savefig('sigma_x.tiff', dpi = 600)
    
    ### plot figure for stress sigma_y
    fig4 = plt.figure(4)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = s22, cmap = 'jet', vmin = -0.1, vmax = 0.4)
    plt.axis('equal')
    plt.colorbar()
    plt.title(r'$\sigma_{y}$')
    plt.savefig('sigma_y.tiff', dpi = 600)
    
    ### plot figure for stress tau_xy
    fig5 = plt.figure(5)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = s12, cmap = 'jet', vmin = -0.1, vmax = 0)
    plt.axis('equal')
    plt.colorbar()
    plt.title(r'$\tau_{xy}$')
    plt.savefig('tau_xy.tiff', dpi = 600)
    
    ### plot figure for hist_loss
    fig6 = plt.figure(6, figsize=(3,3), dpi = 600)
    plt.plot(his_loss[0], color = 'r')
    plt.plot(his_loss[1], color = 'b')
    plt.plot(his_loss[0] + his_loss[1], color = 'k')
    plt.yscale('log')
    plt.xlabel('Iteration', fontdict = {'fontname': 'Helvetica'})
    plt.ylabel('Loss', fontdict = {'fontname': 'Helvetica'})
    plt.title('Loss History', fontdict = {'fontname': 'Helvetica'})
    plt.legend(['$L_{ge}$', '$L_{bc}$', 'L'])
    plt.savefig('hist_loss.tiff', dpi = 600, bbox_inches = 'tight')
    plt.show()
    
    ### output data in the 'out.mat' file
    scipy.io.savemat('out.mat', {'xy': xy, 'u': np.hstack([u,v]), 's11': s11, 's22': s22, 's12': s12}) 
    
    return None