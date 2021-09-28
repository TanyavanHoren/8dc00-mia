"""
Project code for image registration topics.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output


def intensity_based_registration_demo():

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread('../data/image_data/1_1_t1.tif')
    Im = plt.imread('../data/image_data/1_1_t1_d.tif')

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    x = np.array([0., 0., 0.])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)

    # the learning rate
    mu = 0.001

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)

def point_based_reg(filename_1, filename_2, fig_name):

    """"
    inputs:
    filename_1: filename of the fixed image
    filename_2: filename of the moving image
    fig_name: name of the file to which we write the resulting png that shows the similarity
    as a function of the iteration number
    """

    # Select the pair of image slices
    I_path = '../data/image_data/' + filename_1
    Im_path = '../data/image_data/' + filename_2
    I = plt.imread(I_path)
    Im = plt.imread(Im_path)
    X, Xm = util.cpselect(I_path, Im_path)
    X = util.c2h(X)
    Xm = util.c2h(Xm)  # we first need to convert to homogeneous coordinates
    T, E = reg.ls_affine(X, Xm)
    print("The transformation matrix is \n", T)
    print("The registration error is \n", E)
    # apply transformation
    T_Xm_image, T_Xm_coor = reg.image_transform(I, T)
    # plot
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    im1 = ax1.imshow(I)
    ax1.set_title('Fixed image')
    im1_b = ax1.scatter(X[0, :], X[1, :], marker='x', facecolor='red')
    im2 = ax2.imshow(Im)
    ax2.set_title('Moving image')
    im2_b = ax2.scatter(Xm[0, :], Xm[1, :], marker='x', facecolor='red')
    im3 = ax3.imshow(T_Xm_image)
    ax3.set_title('Transformed image')
    im4 = ax4.imshow(I)
    im5 = ax4.imshow(T_Xm_image, alpha=0.7)
    ax4.set_title('Fixed versus transformed')
    plt.savefig('../data/image_data/' + fig_name)
    plt.close(fig)


def ib_reg(filename_1, filename_2, reg_type, metric, mu, num_iter, fig_name, live_plotting):

    """"
    inputs:
    filename_1: filename of the fixed image
    filename_2: filename of the moving image
    reg_type: "rigid" for rigid transformation, "affine" for affine transformation
    metric: "cc" for cross-correlation, "mi" for mutual information
    mu: the learning rate, which determines the step size
    num_iter: the number of iterations
    fig_name: name of the file to which we write the resulting png that shows the similarity
    as a function of the iteration number
    live_plotting: True if we want plots in the notebook, False if not
    """

    # read the fixed and moving images
    I = plt.imread('../data/image_data/' + filename_1)
    Im = plt.imread('../data/image_data/' + filename_2)

    # initial values for the parameters
    # we start with the identity transformation
    if reg_type == "rigid":
        x = np.array([0., 0., 0.])
    elif reg_type == "affine":
        x = np.array([0., 1., 1., 0., 0., 0., 0.])

    # the similarity function
    # this line of code in essence creates a version of rigid_corr(),
    # affine_corr(), rigid_mi() or affine_mi()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    if reg_type == "rigid":
        if metric == "cc":
            fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)
        elif metric == "mi":
            fun = lambda x: reg.rigid_mi(I, Im, x, return_transform=False)
    elif reg_type == "affine":
        if metric == "cc":
            fun = lambda x: reg.affine_corr(I, Im, x, return_transform=False)
        elif metric == "mi":
            fun = lambda x: reg.affine_mi(I, Im, x, return_transform=False)

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    if live_plotting == True:
        fig = plt.figure(figsize=(14,6))

        # fixed and moving image, and parameters
        ax1 = fig.add_subplot(121)

        # fixed image
        im1 = ax1.imshow(I)
        # moving image
        im2 = ax1.imshow(I, alpha=0.7)
        # parameters
        txt = ax1.text(0.3, 0.95,
            np.array2string(x, precision=5, floatmode='fixed'),
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
            transform=ax1.transAxes)

        # 'learning' curve
        ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

        learning_curve, = ax2.plot(iterations, similarity, lw=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Similarity')
        ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):
        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g * mu

        # for visualization of the result
        if reg_type == "rigid":
            if metric == "cc":
                S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True)
            elif metric == "mi":
                S, Im_t, _ = reg.rigid_mi(I, Im, x, return_transform=True)
        elif reg_type == "affine":
            if metric == "cc":
                S, Im_t, _ = reg.affine_corr(I, Im, x, return_transform=True)
            elif metric == "mi":
                S, Im_t, _ = reg.affine_mi(I, Im, x, return_transform=True)

        clear_output(wait=True)

        if live_plotting == True:
            # update moving image and parameters
            im2.set_data(Im_t)
            txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S

        if live_plotting == True:
            learning_curve.set_ydata(similarity)

            display(fig)

    plt.rcParams.update({'font.size': 18})
    fig_save = plt.figure(figsize=(8, 6))
    plt.plot(iterations, similarity, lw=2)
    plt.xlabel('Iteration')
    plt.ylabel('Similarity')
    plt.xlim([0, num_iter])
    plt.ylim([0, 1.2])
    plt.legend(["$\mu$ = " + str(mu)])
    plt.grid()
    plt.savefig('../data/image_data/' + fig_name)
    plt.close(fig_save)


