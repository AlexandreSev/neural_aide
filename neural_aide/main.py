import os
import logging
from os.path import join as pjoin
import datetime
import sys

import pickle
import json
import h5py

import numpy as np
import tensorflow as tf

try:
    from matplotlib import pyplot as plt
    import seaborn as sns
    from pylab import rcParams
    rcParams['figure.figsize'] = 16, 7
except Exception as e:
    pass

import tf_utils as utils
import neural_aide


def create_query(query, data):
    """
    Define scenari.
    """
    if query == 1:
        y = ((data[:, 0] > 190) * (data[:, 0] < 200) * (data[:, 1] > 53) *
             (data[:, 1] < 57)).reshape((-1, 1))
    elif query == 2:
        y = (
            (data[:, 2] > 180) * (data[:, 2] < 210)
            * (data[:, 3] > 50) * (data[:, 3] < 60)
            * ((
                (data[:, 0] - 682.5) ** 2 +
                (data[:, 1] - 1022.5) ** 2
               ) < 90 ** 2)
            ).reshape((-1, 1))
    elif query == 3:
        y = (
            (data[:, 2] > 150) * (data[:, 2] < 240)
            * (data[:, 3] > 40) * (data[:, 3] < 70)
            * ((
                (data[:, 0] - 682.5) ** 2 +
                (data[:, 1] - 1022.5) ** 2
               ) < 280 ** 2)
            * ((data[:, 4] ** 2 + data[:, 5] ** 2) > 0.2 ** 2)
            ).reshape((-1, 1))
    return y


def run_experiment_with_sdss(ressources_folder, qdb=True,
                             random=True, shapes=[32, 32, 1],
                             include_background=True, evolutive_small=True,
                             nb_biased_epoch=2000, use_main_weights=False,
                             display=False, save_plot=False, query=1,
                             biased_lr=0.001, tsm=False, pltlim=True,
                             tsm_lim=None, reduce_factor=None,
                             nb_background_points=None, pool_size=None,
                             np_seed=None, tf_seed=None, saving_dir=None,
                             main_lr=0.001, nn_activation="relu",
                             nn_loss="binary_crossentropy",
                             background_sampling="uncertain", max_iter=11,
                             log_into_file=True):
    """
    Run the active search.
    Params:
        ressources_folder (string): path to the ressources folder with
            databases and results folder.
        qdb (boolean): Use of query by disagreement. If False, it uses
            uncertainty sampling.
        random (boolean): If qdb, choose if a sample if randomly sampled from
            the disagreement area or if it uses a criterion (WIP).
        shapes (list of integers): shapes of the neural network(s) used.
        include_background (boolean): Include background points in the stopping
            criterion of the biased nns. Used only if qdb is set to True.
        evolutive_small (boolean): If True, the number of background points
            will increase with iterations, and the weight of background points
            will always be half of the weight of true points. If False, this
            number will be constant (200), and the weights of background points
            will be equal to the weight of one true sample.
        nb_biased_epoch (integer): Maximum number of epochs in one iteration
            for the biased neural networks.
        use_main_weights (boolean): If True, the biased neural network will use
            the parameters of the main neural network as an initialisation at
            each step. Else, they keep their own parameters during the
            iterations.
        display (boolean): If True, it will display advancement plots.
            Be carefull, the program is pending while the window is opened.
        save_plot (boolean): If True, advancement plot will be saved.
        query (integer): CF create_query function.
        biased_lr (real): learning rate of biased neural networks.
        tsm (boolean): Use or not three set.
        pltlim (boolean): Zoom on the positive area
        tsm_lim (integerr): Maximum number of tuples examined during one
            iteration of tsm.update_regions
        reduce_factor (real or string): The gradients of the biased sample will
            be divided by this factor. If None, it will be equal to
            len(biased_sample). If "evolutive", it will be equal to
            len(biased_samples) * 2. / X_train.shape[0].
        pool_size (integer): Size of the pool considered to find the most
            uncertain point. If None, the whole X is used.
        np_seed (int): Seed used by numpy. Can be None.
        tf_seed (int): Seed used by tensorflow. Can be None.
        saving_dir (string): where to save the results. If None, it will
            be in ressources_folder/results.
        main_lr (real): learning rate of the main model
        nn_activation (string): activation function of the neural networks
        nn_loss (string): loss of the neural networks
        background_sampling (string). If "uncertain", background points will be
            the most uncertain of the model. If "random", background points
            will be randomly sampled.
    """

    if saving_dir is None:
        SAVING_DIRECTORY = pjoin(ressources_folder, "results")
    else:
        SAVING_DIRECTORY = saving_dir

    # Create unique identifier
    date = str(datetime.datetime.now())
    date = date.replace("-", "_").replace(" ", "_").replace(":", "_")

    identifier = "sdss_" + date
    for name, value in zip([
            "qdb", "random", "shapes", "include_background", "evolutive_small",
            "nb_background_points", "nb_biased_epoch", "use_main_weights",
            "query", "tsm", "biased_lr", "reduce_factor", "pool_size",
                            ], [
            qdb, random, shapes, include_background, evolutive_small,
            nb_background_points, nb_biased_epoch, use_main_weights, query,
            tsm, biased_lr, reduce_factor, pool_size
                               ]):

        identifier += "_" + name + "_" + str(value)

    # Create saving directory
    SAVING_DIRECTORY = pjoin(SAVING_DIRECTORY, identifier)
    os.mkdir(SAVING_DIRECTORY)

    # Configurate logger
    logging.shutdown()
    reload(logging)
    if log_into_file:
        log_file = (pjoin(SAVING_DIRECTORY, "log.log"))
    else:
        log_file = None
    utils.initialize_logger(log_file, filemode="w")

    # Fix the seeds
    if np_seed is None:
        seed = np.random.randint(1000)
    else:
        seed = np_seed
    np.random.seed(seed)
    logging.info("Numpy seed: %s" % seed)

    if tf_seed is None:
        seed = np.random.randint(1000)
    else:
        seed = tf_seed
    tf.set_random_seed(seed)
    logging.info("Tensorflow seed: %s" % seed)

    for name, value in zip([
            "qdb", "random", "shapes", "include_background", "evolutive_small",
            "nb_biased_epoch", "use_main_weights", "query", "tsm", "biased_lr",
            "reduce_factor", "pool_size",
                            ], [
            qdb, random, shapes, include_background, evolutive_small,
            nb_biased_epoch, use_main_weights, query, tsm, biased_lr,
            reduce_factor, pool_size
                               ]):
        logging.info("%s: %s" % (name, value))

    # Load data
    if query == 1:
        data_file = "sdss_ra_dec_all.npy"
    elif query == 2:
        data_file = "sdss_rowc_colc_ra_dec_all.npy"
    elif query == 3:
        data_file = "sdss_rowc_colc_ra_dec_rowv_colv_all.npy"
    else:
        raise ValueError("Query not understood")

    DATAPATH = pjoin(ressources_folder, "ressources", "sdss", data_file)

    data = np.load(DATAPATH)
    logging.info("Data Shape: " + str(data.shape))

    X = neural_aide.utils_database.normalize_npy_database(data)

    # Creation of the labels
    y = create_query(query, data)

    logging.info("Selectivity of the query: %s" % np.mean(y))

    x_lb = np.min(X[(y == 1).reshape(-1), 0])
    x_ub = np.max(X[(y == 1).reshape(-1), 0])
    y_lb = np.min(X[(y == 1).reshape(-1), 1])
    y_ub = np.max(X[(y == 1).reshape(-1), 1])

    # Limit of plots
    if pltlim:
        xlim = (0.9 * x_lb, 1.1 * x_ub)
        ylim = (0.9 * y_lb, 1.1 * y_ub)
    else:
        xlim, ylim = None, None

    main_weights_path = pjoin(SAVING_DIRECTORY, "weights_main.pckl")

    if save_plot:
        plot_save_path = pjoin(SAVING_DIRECTORY, "plots/plot_%s.png")

        os.mkdir("/".join(plot_save_path.split("/")[: -1]))
    else:
        plot_save_path = None

    callback_save_path = pjoin(SAVING_DIRECTORY, "callback.pckl")

    timer_save_path = pjoin(SAVING_DIRECTORY, "timer.pckl")

    pos_save_path = pjoin(SAVING_DIRECTORY, "weights_pos.pckl")

    neg_save_path = pjoin(SAVING_DIRECTORY, "weights_neg.pckl")

    if use_main_weights:
        neural_aide.active_search.active_search(
            X, y, shapes=shapes, max_iterations=max_iter,
            pos_weights_path=main_weights_path,
            neg_weights_path=main_weights_path,
            main_weights_path=main_weights_path,
            display_plot=display, plot_save_path=plot_save_path,
            n_points=10000, callback_save_path=callback_save_path,
            nb_max_main_epoch=4000, qdb=qdb, random=random, xlim=xlim,
            ylim=ylim, timer_save_path=timer_save_path, save_biased=False,
            include_background=include_background,
            evolutive_small=evolutive_small,
            nb_background_points=nb_background_points,
            nb_biased_epoch=nb_biased_epoch,
            biased_lr=biased_lr, tsm=tsm, tsm_lim=tsm_lim,
            reduce_factor=reduce_factor, pool_size=pool_size, main_lr=main_lr,
            nn_activation=nn_activation, nn_loss=nn_loss,
            background_sampling=background_sampling
        )
    else:
        neural_aide.active_search.active_search(
            X, y, shapes=shapes, max_iterations=max_iter,
            pos_weights_path=pos_save_path,
            neg_weights_path=neg_save_path,
            main_weights_path=main_weights_path,
            display_plot=display, plot_save_path=plot_save_path,
            n_points=10000, callback_save_path=callback_save_path,
            nb_max_main_epoch=4000, qdb=qdb, random=random, xlim=xlim,
            ylim=ylim, timer_save_path=timer_save_path, save_biased=True,
            include_background=include_background,
            evolutive_small=evolutive_small,
            nb_background_points=nb_background_points,
            nb_biased_epoch=nb_biased_epoch,
            biased_lr=biased_lr, tsm=tsm, tsm_lim=tsm_lim,
            reduce_factor=reduce_factor, pool_size=pool_size, main_lr=main_lr,
            nn_activation=nn_activation, nn_loss=nn_loss,
            background_sampling=background_sampling
        )


def run_experiment_with_housing(ressources_folder, qdb=True,
                                random=True, shapes=[32, 32, 1],
                                include_background=True, evolutive_small=True,
                                nb_biased_epoch=2000, use_main_weights=False,
                                biased_lr=0.001, tsm=False,
                                tsm_lim=None, reduce_factor=None,
                                pool_size=None, khaled=False, np_seed=None,
                                tf_seed=None, main_lr=0.001,
                                nn_activation="relu",
                                nn_loss="binary_crossentropy",
                                background_sampling="uncertain", max_iter=11):
    """
    Run the active search.
    Params:
        ressources_folder (string): path to the ressources folder with
            databases and results folder.
        qdb (boolean): Use of query by disagreement. If False, it uses
            uncertainty sampling.
        random (boolean): If qdb, choose if a sample if randomly sampled from
            the disagreement area or if it uses a criterion (WIP).
        shapes (list of integers): shapes of the neural network(s) used.
        include_background (boolean): Include background points in the stopping
            criterion of the biased nns. Used only if qdb is set to True.
        evolutive_small (boolean): If True, the number of background points
            will increase with iterations, and the weight of background points
            will always be half of the weight of true points. If False, this
            number will be constant (200), and the weights of background points
            will be equal to the weight of one true sample.
        nb_biased_epoch (integer): Maximum number of epochs in one iteration
            for the biased neural networks.
        use_main_weights (boolean): If True, the biased neural network will use
            the parameters of the main neural network as an initialisation at
            each step. Else, they keep their own parameters during the
            iterations.
        biased_lr (real): learning rate of biased neural networks.
        tsm (boolean): Use or not three set.
        tsm_lim (integerr): Maximum number of tuples examined during one
            iteration of tsm.update_regions
        reduce_factor (real or string): The gradients of the biased sample will
            be divided by this factor. If None, it will be equal to
            len(biased_sample). If "evolutive", it will be equal to
            len(biased_samples) * 2. / X_train.shape[0].
        pool_size (integer): Size of the pool considered to find the most
            uncertain point. If None, the whole X is used.
        khaled (boolean): Decide if the nn is trained with alex' or khaled's
            labels.
        np_seed (int): Seed used by numpy. Can be None.
        tf_seed (int): Seed used by tensorflow. Can be None.
        main_lr (real): learning rate of the main model
        nn_activation (string): activation function of the neural network.
        nn_loss (string): loss of the neural networks
        background_sampling (string). If "uncertain", background points will be
            the most uncertain of the model. If "random", background points
            will be randomly sampled.
    """

    SAVING_DIRECTORY = pjoin(ressources_folder, "results")

    # Create unique identifier
    date = str(datetime.datetime.now())
    date = date.replace("-", "_").replace(" ", "_").replace(":", "_")

    identifier = "housing" + date
    for name, value in zip([
            "qdb", "random", "shapes", "include_background", "evolutive_small",
            "nb_biased_epoch", "use_main_weights", "tsm", "biased_lr",
            "reduce_factor", "pool_size",
                            ], [
            qdb, random, shapes, include_background, evolutive_small,
            nb_biased_epoch, use_main_weights, tsm, biased_lr,
            reduce_factor, pool_size,
                               ]):

        identifier += "_" + name + "_" + str(value)

    # Create saving directory
    SAVING_DIRECTORY = pjoin(SAVING_DIRECTORY, identifier)
    os.mkdir(SAVING_DIRECTORY)

    # Configurate logger
    logging.shutdown()
    reload(logging)
    log_file = (pjoin(SAVING_DIRECTORY, "log.log"))
    utils.initialize_logger(log_file, filemode="w")

    # Fix the seeds
    if np_seed is None:
        seed = np.random.randint(1000)
    else:
        seed = np_seed
    np.random.seed(seed)
    logging.info("Seed: %s" % seed)

    if tf_seed is None:
        seed = np.random.randint(1000)
    else:
        seed = np_seed
    tf.set_random_seed(seed)
    logging.info("Tensorflow seed: %s" % seed)

    for name, value in zip([
            "qdb", "random", "shapes", "include_background", "evolutive_small",
            "nb_biased_epoch", "use_main_weights", "tsm", "biased_lr",
            "reduce_factor", "pool_size",
                            ], [
            qdb, random, shapes, include_background, evolutive_small,
            nb_biased_epoch, use_main_weights, tsm, biased_lr,
            reduce_factor, pool_size,
                               ]):
        logging.info("%s: %s" % (name, value))

    # Load data

    DATAPATH = pjoin(ressources_folder, "ressources", "housing_dataset",
                     "sorted_img_emb.h5")

    h5f = h5py.File(DATAPATH, "r")
    data = h5f["img_emb"][:]
    h5f.close()

    logging.info("Data Shape: " + str(data.shape))

    X = neural_aide.utils_database.normalize_npy_database(data)

    # Creation of the labels
    if khaled:
        logging.info("Train on Khaled' labels")
        with open(pjoin(ressources_folder, "ressources", "housing_dataset",
                        "khaled_labels.json"), "r") as fp:
            y = np.array(json.load(fp)).reshape((-1, 1))
    else:
        logging.info("Train on Alex' labels")
        with open(pjoin(ressources_folder, "ressources", "housing_dataset",
                        "alex_labels.json"), "r") as fp:
            y = np.array(json.load(fp)).reshape((-1, 1))

    logging.info("Selectivity of the query: %s" % np.mean(y))

    main_weights_path = pjoin(SAVING_DIRECTORY, "weights_main.pckl")

    callback_save_path = pjoin(SAVING_DIRECTORY, "callback.pckl")

    timer_save_path = pjoin(SAVING_DIRECTORY, "timer.pckl")

    pos_save_path = pjoin(SAVING_DIRECTORY, "weights_pos.pckl")

    neg_save_path = pjoin(SAVING_DIRECTORY, "weights_neg.pckl")

    if use_main_weights:
        neural_aide.active_search.active_search(
            X, y, shapes=shapes, max_iterations=max_iter,
            pos_weights_path=main_weights_path,
            neg_weights_path=main_weights_path,
            main_weights_path=main_weights_path,
            display_plot=False, plot_save_path=None,
            n_points=10000, callback_save_path=callback_save_path,
            nb_max_main_epoch=4000, qdb=qdb, random=random, xlim=None,
            ylim=None, timer_save_path=timer_save_path, save_biased=False,
            include_background=include_background,
            evolutive_small=evolutive_small, nb_biased_epoch=nb_biased_epoch,
            biased_lr=biased_lr, tsm=tsm, tsm_lim=tsm_lim,
            reduce_factor=reduce_factor, main_lr=main_lr,
            nn_activation=nn_activation, nn_loss=nn_loss,
            background_sampling=background_sampling
        )
    else:
        neural_aide.active_search.active_search(
            X, y, shapes=shapes, max_iterations=max_iter,
            pos_weights_path=pos_save_path,
            neg_weights_path=neg_save_path,
            main_weights_path=main_weights_path,
            display_plot=False, plot_save_path=None,
            n_points=10000, callback_save_path=callback_save_path,
            nb_max_main_epoch=4000, qdb=qdb, random=random, xlim=None,
            ylim=None, timer_save_path=timer_save_path, save_biased=True,
            include_background=include_background,
            evolutive_small=evolutive_small, nb_biased_epoch=nb_biased_epoch,
            biased_lr=biased_lr, tsm=tsm, tsm_lim=tsm_lim,
            reduce_factor=reduce_factor, main_lr=main_lr,
            nn_activation=nn_activation, n_loss=nn_loss,
            background_sampling=background_sampling
        )
