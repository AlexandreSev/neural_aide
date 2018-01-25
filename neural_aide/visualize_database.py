#!/usr/bin/python
# coding: utf-8

import logging
import numpy as np
try:
    from matplotlib import pyplot as plt
    import seaborn as sns
except Exception as e:
    pass


def visualize_npy_database(data, xindex=0, yindex=1, savepath=None):
    """
    Visualiza a dataset
    Params:
        data (np.array): Data to plot.
        xindex (integer): Column to use as x.
        yindex (integer): Column to use as y.
        savepath (string): Where to save the plot.
                           If None, the plot is not saved.
    """

    plt.figure()
    plt.scatter(data[:, xindex], data[:, yindex])

    if savepath is not None:
        plt.savefig(savepath)

    plt.show()
    plt.close()


def visualize_npy_file_database(file, xindex=0, yindex=1, savepath=None):
    """
    Visualiza a dataset
    Params:
        file (string): File with data to plot.
        xindex (integer): Column to use as x.
        yindex (integer): Column to use as y.
        savepath (string): Where to save the plot.
                           If None, the plot is not saved.
    """

    data = np.load(file)

    visualize_npy_database(data, xindex, yindex, savepath)


def plot_filters(xs, ys, path):
    """
    Plot the filters of a nn
    Params:
        xs (list of integers): list of x coordinate for points on which the
            filters are evaluated.
        ys (list of integers): list of y coordinate for points on which the
            filters are evaluated.
        path (string): Path of the file where are saved the weights of the
            neural network
    """
    with open(path, "r") as fp:
        dico = pickle.load(fp)

    W = dico["weights"]["W0"]
    b = dico["biases"]["b0"]

    n_filters = W.shape[1]
    colors = ["red", "green"]

    plt.figure(1)

    for i in range(n_filters):
        colors_temp = [W[0, i] * x + W[1, i] * y + b[i] > 0
                       for x, y in zip(xs, ys)]
        colors_temp = [colors[j] for j in colors_temp]

        plt.subplot(101 + n_filters * 10 + i)
        plt.scatter(xs, ys, c=colors_temp)
        plt.grid(True)
    plt.show()


def plot_advancement_uncertainty_search(X_train, y_train, X_val, y_val,
                                        old_pred, new_pred, save_path=None,
                                        show=False, xlim=None, ylim=None):
    """
    Plot 3 figures to vizualize the progress of the uncertainty search.
    Params:
        X (np.array): Data to visualize.
        y (np.array): True labels
        old_pred (np.array): Labels given at the last iteration.
        new_pred (np.array): Labels given at the current iteration.
        save_path (string): where to save the plot. If None, the plot is not
            save.
        show (boolean): if True, display the graphique.
        xlim (2-uple of integers): limits of the x axis.
        ylim (2-uple of integers): limits of the y axis.
    """
    colors = ["red", "green"]

    # If xlim/ylim is not, find the region of interest:
    if xlim is None:
        xlow = np.min(X_val[y_val.reshape(-1) > 0.5, 0]),
                    
        xup = np.max(X_val[y_val.reshape(-1) > 0.5, 0]),     

        if np.where(X_val, old_pred.reshape(-1) > 0.5):
            xlow = min(xlow, np.min(X_val[old_pred.reshape(-1) > 0.5, 0]))
            xup = max(xup, np.max(X_val[old_pred.reshape(-1) > 0.5, 0]))

        if np.where(X_val, new_pred.reshape(-1) > 0.5):
            xlow = min(xlow, np.min(X_val[new_pred.reshape(-1) > 0.5, 0]))
            xup = max(xup, np.max(X_val[new_pred.reshape(-1) > 0.5, 0]))

        delta = xup - xlow
        xlim = (xlow - delta / 20, xup + delta / 20)

    if ylim is None:
        ylow = np.min(X_val[y_val.reshape(-1) > 0.5, 1])
        yup = np.max(X_val[y_val.reshape(-1) > 0.5, 1]),     

        if np.where(X_val, old_pred.reshape(-1) > 0.5):
            ylow = min(ylow, np.min(X_val[old_pred.reshape(-1) > 0.5, 1]))
            yup = max(yup, np.max(X_val[old_pred.reshape(-1) > 0.5, 1]))

        if np.where(X_val, new_pred.reshape(-1) > 0.5):
            ylow = min(ylow, np.min(X_val[new_pred.reshape(-1) > 0.5, 1]))
            yup = max(yup, np.max(X_val[new_pred.reshape(-1) > 0.5, 1]))
            
        delta = yup - ylow
        ylim = (ylow - delta / 20, yup + delta / 20)

    plt.figure(1)
    # Plot the old predictions with uncertain points
    plt.subplot(131)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(X_val[:, 0], X_val[:, 1],
                c=[colors[int(j > 0.5)] for j in old_pred])

    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=["grey" for j in range(X_train.shape[0]-1)] + ["black"])

    plt.grid(True)

    plt.subplot(132)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(X_val[:, 0], X_val[:, 1],
                c=[colors[int(j > 0.5)] for j in new_pred])

    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=["grey" for j in range(X_train.shape[0]-1)] + ["black"])

    plt.grid(True)

    plt.subplot(133)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(X_val[:, 0], X_val[:, 1],
                c=[colors[int(k)] for k in y_val.reshape(-1)])

    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=["grey" for j in range(X_train.shape[0]-1)] + ["black"])
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.clf()


def random_uncertainty_plot(X_train, y_train, X_val, y_val, old_pred, new_pred,
                            n_points=10000, save_path=None, show=False,
                            xlim=None, ylim=None):
    """
    Plot 4 figures with random pointsto vizualize the progress of the qdb
    search.
    Params:
        X (np.array): Data to visualize.
        y (np.array): True labels.
        old_pred (np.array): Labels given at the last iteration.
        new_pred (np.array): Labels given at the current iteration.
        n_points (integer): number of points to plot.
        save_path (string): where to save the plot. If None, the plot is not
            save.
        show (boolean): if True, display the graphique.
        xlim (2-uple of integers): limits of the x axis.
        ylim (2-uple of integers): limits of the y axis.
    """
    if (((xlim is None) and (ylim is not None))
            or ((ylim is None) and (xlim is not None))):
        raise Exception("Not implemented yet !")

    if xlim is not None:
        availablelity_filter = (
            (X_val[:, 0] > xlim[0])
            * (X_val[:, 0] < xlim[1])
            * (X_val[:, 1] > ylim[0])
            * (X_val[:, 1] < ylim[1])
            )
        available_points = np.where(availablelity_filter == 1)[0]
    else:
        available_points = np.ones(y_val.shape[0]).astype(bool)

    order = np.arange(y_val[available_points].shape[0])
    np.random.shuffle(order)
    order = order[:min(n_points, y_val[available_points].shape[0])]

    X_to_plot = X_val[available_points][order]
    y_to_plot = y_val[available_points][order]
    old_pred_to_plot = old_pred[available_points][order]
    new_pred_to_plot = new_pred[available_points][order]

    plot_advancement_uncertainty_search(
        X_train, y_train, X_to_plot, y_to_plot, old_pred_to_plot,
        new_pred_to_plot, save_path, show, xlim, ylim
        )


def plot_advancement_qdb_search(X_train, y_train, X_val, y_val, old_pred,
                                new_pred, pred_pos, pred_neg,
                                uncertain_samples, save_path=None, show=False,
                                xlim=None, ylim=None):
    """
    Plot 4 figures to vizualize the progress of the qdb search.
    Params:
        X (np.array): Data to visualize.
        y (np.array): True labels
        old_pred (np.array): Labels given at the last iteration.
        new_pred (np.array): Labels given at the current iteration.
        pred_pos (np.array): Labels given by the positevely biased model.
        pred_neg (np.array): Labels given by the negatively biased model.
        samples (list of integers): Indices of labeled samples.
        uncertain_samples (list of integers): Indices of samples used to train
            biased nn.
        save_path (string): where to save the plot. If None, the plot is not
            save.
        show (boolean): if True, display the graphique.
        xlim (2-uple of integers): limits of the x axis.
        ylim (2-uple of integers): limits of the y axis.
    """

    colors = ["red", "green"]
    colors_others = ["red", "yellow", "green"]

    # If xlim/ylim is not, find the region of interest:
    if xlim is None:
        xlow = min((np.min(X_val[y_val.reshape(-1) > 0.5, 0]),
                    np.min(X_val[(
                        (pred_pos.reshape(-1) > 0.5) +
                        (pred_neg.reshape(-1) > 0.5)
                        ) == 1, 0]),
                    ))

        xup = max((np.max(X_val[y_val.reshape(-1) > 0.5, 0]),
                   np.max(X_val[(
                        (pred_pos.reshape(-1) > 0.5) +
                        (pred_neg.reshape(-1) > 0.5)
                        ) == 1, 0]),
                    ))

        if np.where(X_val, old_pred.reshape(-1) > 0.5):
            xlow = min(xlow, np.min(X_val[old_pred.reshape(-1) > 0.5, 0]))
            xup = max(xup, np.max(X_val[old_pred.reshape(-1) > 0.5, 0]))

        if np.where(X_val, new_pred.reshape(-1) > 0.5):
            xlow = min(xlow, np.min(X_val[new_pred.reshape(-1) > 0.5, 0]))
            xup = max(xup, np.max(X_val[new_pred.reshape(-1) > 0.5, 0]))

        delta = xup - xlow
        xlim = (xlow - delta / 20, xup + delta / 20)

    if ylim is None:
        ylow = min((np.min(X_val[y_val.reshape(-1) > 0.5, 1]),
                    np.min(X_val[(
                        (pred_pos.reshape(-1) > 0.5) +
                        (pred_neg.reshape(-1) > 0.5)
                        ) == 1, 1]),
                    ))
        yup = max((np.max(X_val[y_val.reshape(-1) > 0.5, 1]),
                   np.max(X_val[(
                        (pred_pos.reshape(-1) > 0.5) +
                        (pred_neg.reshape(-1) > 0.5)
                        ) == 1, 1]),
                    ))

        if np.where(X_val, old_pred.reshape(-1) > 0.5):
            ylow = min(ylow, np.min(X_val[old_pred.reshape(-1) > 0.5, 1]))
            yup = max(yup, np.max(X_val[old_pred.reshape(-1) > 0.5, 1]))

        if np.where(X_val, new_pred.reshape(-1) > 0.5):
            ylow = min(ylow, np.min(X_val[new_pred.reshape(-1) > 0.5, 1]))
            yup = max(yup, np.max(X_val[new_pred.reshape(-1) > 0.5, 1]))
            
        delta = yup - ylow
        ylim = (ylow - delta / 20, yup + delta / 20)

    plt.figure(1)
    # Plot the old predictions with uncertain points
    plt.subplot(141)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(X_val[:, 0], X_val[:, 1],
                c=[colors[int(j > 0.5)] for j in old_pred])

    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=["grey" for j in range(X_train.shape[0]-1)] + ["black"])

    plt.scatter(X_val[uncertain_samples, 0], X_val[uncertain_samples, 1],
                c=["orange" for j in range(len(uncertain_samples))])
    plt.grid(True)

    # Plot the biased prediction
    plt.subplot(142)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(X_val[:, 0], X_val[:, 1],
                c=[colors_others[int(k > 0.5) + int(j > 0.5)]
                   for k, j in zip(pred_pos, pred_neg)])

    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=["grey" for j in range(X_train.shape[0]-1)] + ["black"])
    plt.grid(True)

    # Plot the new prediction
    plt.subplot(143)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(X_val[:, 0], X_val[:, 1],
                c=[colors[int(j > 0.5)] for j in new_pred])

    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=["grey" for j in range(X_train.shape[0]-1)] + ["black"])
    plt.grid(True)

    # Plot the ground truth
    plt.subplot(144)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(X_val[:, 0], X_val[:, 1],
                c=[colors[int(k)] for k in y_val.reshape(-1)])

    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=["grey" for j in range(X_train.shape[0]-1)] + ["black"])
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.clf()


def random_advancement_plot(X_train, y_train, X_val, y_val, old_pred, new_pred,
                            pred_pos, pred_neg, uncertain_samples,
                            n_points=10000, save_path=None, show=False,
                            xlim=None, ylim=None):
    """
    Plot 4 figures with random pointsto vizualize the progress of the qdb
    search.
    Params:
        X (np.array): Data to visualize.
        y (np.array): True labels.
        old_pred (np.array): Labels given at the last iteration.
        new_pred (np.array): Labels given at the current iteration.
        pred_pos (np.array): Labels given by the positevely biased model.
        pred_neg (np.array): Labels given by the negatively biased model.
        samples (list of integers): Indices of labeled samples.
        uncertain_samples (list of integers): Indices of samples used to train
            biased nn.
        n_points (integer): number of points to plot.
        save_path (string): where to save the plot. If None, the plot is not
            save.
        show (boolean): if True, display the graphique.
        xlim (2-uple of integers): limits of the x axis.
        ylim (2-uple of integers): limits of the y axis.
    """
    if (((xlim is None) and (ylim is not None))
            or ((ylim is None) and (xlim is not None))):
        raise Exception("Not implemented yet !")

    if xlim is None:
        xlow = min((np.min(X_val[y_val.reshape(-1) > 0.5, 0]),
                    np.min(X_val[(
                        (pred_pos.reshape(-1) > 0.5) +
                        (pred_neg.reshape(-1) > 0.5)
                        ) == 1, 0]),
                    ))

        xup = max((np.max(X_val[y_val.reshape(-1) > 0.5, 0]),
                   np.max(X_val[(
                        (pred_pos.reshape(-1) > 0.5) +
                        (pred_neg.reshape(-1) > 0.5)
                        ) == 1, 0]),
                    ))

        if np.sum(old_pred.reshape(-1) > 0.5):
            xlow = min(xlow, np.min(X_val[old_pred.reshape(-1) > 0.5, 0]))
            xup = max(xup, np.max(X_val[old_pred.reshape(-1) > 0.5, 0]))

        if np.sum(new_pred.reshape(-1) > 0.5):
            xlow = min(xlow, np.min(X_val[new_pred.reshape(-1) > 0.5, 0]))
            xup = max(xup, np.max(X_val[new_pred.reshape(-1) > 0.5, 0]))

        delta = xup - xlow
        xlim = (xlow - delta / 20, xup + delta / 20)

    if ylim is None:
        ylow = min((np.min(X_val[y_val.reshape(-1) > 0.5, 1]),
                    np.min(X_val[(
                        (pred_pos.reshape(-1) > 0.5) +
                        (pred_neg.reshape(-1) > 0.5)
                        ) == 1, 1]),
                    ))
        yup = max((np.max(X_val[y_val.reshape(-1) > 0.5, 1]),
                   np.max(X_val[(
                        (pred_pos.reshape(-1) > 0.5) +
                        (pred_neg.reshape(-1) > 0.5)
                        ) == 1, 1]),
                    ))

        if np.sum(old_pred.reshape(-1) > 0.5):
            ylow = min(ylow, np.min(X_val[old_pred.reshape(-1) > 0.5, 1]))
            yup = max(yup, np.max(X_val[old_pred.reshape(-1) > 0.5, 1]))

        if np.sum(new_pred.reshape(-1) > 0.5):
            ylow = min(ylow, np.min(X_val[new_pred.reshape(-1) > 0.5, 1]))
            yup = max(yup, np.max(X_val[new_pred.reshape(-1) > 0.5, 1]))
            
        delta = yup - ylow
        ylim = (ylow - delta / 20, yup + delta / 20)

    availablelity_filter = (
        (X_val[:, 0] > xlim[0])
        * (X_val[:, 0] < xlim[1])
        * (X_val[:, 1] > ylim[0])
        * (X_val[:, 1] < ylim[1])
        )
    available_points = np.where(availablelity_filter == 1)[0]

    order = np.arange(y_val[available_points].shape[0])
    np.random.shuffle(order)
    order = order[:min(n_points, y_val[available_points].shape[0])]

    logging.debug("uncertain_samples: %s" % uncertain_samples)

    X_to_plot = np.vstack((
        X_val[uncertain_samples],
        X_val[available_points][order]
        ))
    y_to_plot = np.vstack((
        y_val[uncertain_samples],
        y_val[available_points][order]
        ))
    old_pred_to_plot = np.vstack((
        old_pred[uncertain_samples],
        old_pred[available_points][order]
        ))
    new_pred_to_plot = np.vstack((
        new_pred[uncertain_samples],
        new_pred[available_points][order]
        ))
    pred_pos_to_plot = np.vstack((
        pred_pos[uncertain_samples],
        pred_pos[available_points][order]
        ))
    pred_neg_to_plot = np.vstack((
        pred_neg[uncertain_samples],
        pred_neg[available_points][order]
        ))
    uncertain_samples_to_plot = range(len(uncertain_samples))

    plot_advancement_qdb_search(
        X_train, y_train, X_to_plot, y_to_plot, old_pred_to_plot,
        new_pred_to_plot, pred_pos_to_plot, pred_neg_to_plot,
        uncertain_samples_to_plot, save_path, show, xlim, ylim
        )


if __name__ == "__main__":
    import argparse

    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='Where to find the data to plot')
    parser.add_argument('--savepath', help="Where to save the plot",
                        default=None)
    parser.add_argument('--xindex', type=int, default=0,
                        help="Column index to use as x")
    parser.add_argument('--yindex', type=int, default=1,
                        help="Column index to use as y")

    # Read the arguments
    args = parser.parse_args()

    visualize_npy_file_database(args.filepath, xindex=args.xindex,
                                yindex=args.yindex, savepath=args.savepath)
