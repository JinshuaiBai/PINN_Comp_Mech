import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def Post_Process(net_u, net_v, net_w, pinn, his_loss):
    """
    ====================================================================================================================

    Post_Process function is to:
        1. Visualize the displacement, strain, and stress;
        2. Output results.

    --------------------------------------------------------------------------------------------------------------------

    Name        Type                    Info.
    
    [u]         [array of float]        : Displacement in x direction;
    [v]         [array of float]        : Displacement in y direction;
    [w]         [array of float]        : Displacement in z direction;
    [s11]       [array of float]        : Normal Stress in x direction;
    [s22]       [array of float]        : Normal Stress in y direction;
    [s33]       [array of float]        : Normal Stress in z direction;
    [s12]       [array of float]        : Shear Stress on xy plane;
    [s23]       [array of float]        : Shear Stress on yz plane;
    [s13]       [array of float]        : Shear Stress on xz plane;
    [net_u]     [keras model]           : The trained FNN for displacement u;
    [net_v]     [keras model]           : The trained FNN for displacement v;
    [net_w]     [keras model]           : The trained FNN for displacement w;
    [pinn]      [Keras model]           : The Physics-Informed Neural Network;
    [his_loss]  [list]                  : History values of the loss terms.

    ====================================================================================================================
    """

    C = scipy.io.loadmat('FEA.mat')
    x = C['X']

    u = net_u.predict(x) * x[..., 0, np.newaxis]
    v = net_v.predict(x) * x[..., 1, np.newaxis]
    w = net_w.predict(x) * x[..., 2, np.newaxis]
    temp = pinn.predict([x for i in range(0, 7)])
    s1 = temp[3]
    s2 = temp[8]
    s3 = temp[13]
    s12 = temp[4]
    s13 = temp[14]
    s23 = temp[10]
    
    ### Visualisation
    font = {'fontname': 'Times new roman'}
    vmax = np.max(u[:, 0])
    vmin = np.min(u[:, 0])
    vlim = np.linspace(vmin, vmax, 10, endpoint=True)
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(projection='3d')
    im = ax.scatter(x[:, 1], x[:, 0], x[:, 2], c=u[:, 0], cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('y [m]', **font, fontsize=20)
    ax.set_ylabel('x [m]', **font, fontsize=20)
    ax.set_zlabel('z [m]', **font, fontsize=20)
    ax.set_title('U', **font, fontsize=24, fontweight='bold')
    ax.set_box_aspect((np.ptp(x[:, 1]), np.ptp(x[:, 0]), np.ptp(x[:, 2])))
    ax.invert_xaxis()
    plt.colorbar(im, ticks=vlim, ax=ax, format='%.3f', fraction=0.02)
    
    vmax = np.max(v[:, 0])
    vmin = np.min(v[:, 0])
    vlim = np.linspace(vmin, vmax, 10, endpoint=True)
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(projection='3d')
    im = ax.scatter(x[:, 1], x[:, 0], x[:, 2], c=v[:, 0], cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('y [m]', **font, fontsize=20)
    ax.set_ylabel('x [m]', **font, fontsize=20)
    ax.set_zlabel('z [m]', **font, fontsize=20)
    ax.set_title('V', **font, fontsize=24, fontweight='bold')
    ax.set_box_aspect((np.ptp(x[:, 1]), np.ptp(x[:, 0]), np.ptp(x[:, 2])))
    ax.invert_xaxis()
    plt.colorbar(im, ticks=vlim, ax=ax, format='%.3f', fraction=0.02)
    
    vmax = np.max(w[:, 0])
    vmin = np.min(w[:, 0])
    vlim = np.linspace(vmin, vmax, 10, endpoint=True)
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(projection='3d')
    im = ax.scatter(x[:, 1], x[:, 0], x[:, 2], c=w[:, 0], cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('y [m]', **font, fontsize=20)
    ax.set_ylabel('x [m]', **font, fontsize=20)
    ax.set_zlabel('z [m]', **font, fontsize=20)
    ax.set_title('W', **font, fontsize=24, fontweight='bold')
    ax.set_box_aspect((np.ptp(x[:, 1]), np.ptp(x[:, 0]), np.ptp(x[:, 2])))
    ax.invert_xaxis()
    plt.colorbar(im, ticks=vlim, ax=ax, format='%.3f', fraction=0.02)
    
    vmax = np.max(s1)
    vmin = np.min(s1)
    vlim = np.linspace(vmin, vmax, 10, endpoint=True)
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(projection='3d')
    im = ax.scatter(x[:, 1], x[:, 0], x[:, 2], c=s1, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('y [m]', **font, fontsize=20)
    ax.set_ylabel('x [m]', **font, fontsize=20)
    ax.set_zlabel('z [m]', **font, fontsize=20)
    ax.set_title(r'$\sigma_x$', **font, fontsize=24, fontweight='bold')
    ax.set_box_aspect((np.ptp(x[:, 1]), np.ptp(x[:, 0]), np.ptp(x[:, 2])))
    ax.invert_xaxis()
    plt.colorbar(im, ticks=vlim, ax=ax, format='%.3f', fraction=0.02)
    
    vmax = np.max(s2)
    vmin = np.min(s2)
    vlim = np.linspace(vmin, vmax, 10, endpoint=True)
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(projection='3d')
    im = ax.scatter(x[:, 1], x[:, 0], x[:, 2], c=s2, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('y [m]', **font, fontsize=20)
    ax.set_ylabel('x [m]', **font, fontsize=20)
    ax.set_zlabel('z [m]', **font, fontsize=20)
    ax.set_title(r'$\sigma_y$', **font, fontsize=24, fontweight='bold')
    ax.set_box_aspect((np.ptp(x[:, 1]), np.ptp(x[:, 0]), np.ptp(x[:, 2])))
    ax.invert_xaxis()
    plt.colorbar(im, ticks=vlim, ax=ax, format='%.3f', fraction=0.02)
    
    vmax = np.max(s3)
    vmin = np.min(s3)
    vlim = np.linspace(vmin, vmax, 10, endpoint=True)
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(projection='3d')
    im = ax.scatter(x[:, 1], x[:, 0], x[:, 2], c=s3, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('y [m]', **font, fontsize=20)
    ax.set_ylabel('x [m]', **font, fontsize=20)
    ax.set_zlabel('z [m]', **font, fontsize=20)
    ax.set_title(r'$\sigma_z$', **font, fontsize=24, fontweight='bold')
    ax.set_box_aspect((np.ptp(x[:, 1]), np.ptp(x[:, 0]), np.ptp(x[:, 2])))
    ax.invert_xaxis()
    plt.colorbar(im, ticks=vlim, ax=ax, format='%.3f', fraction=0.02)
    
    scipy.io.savemat('out.mat', {'x': x, 'u': np.hstack([u, v, w]), 's1': s1, 's2': s2, 's3': s3, 's12': s12, 's23': s23, 's13': s13})
    
    return None