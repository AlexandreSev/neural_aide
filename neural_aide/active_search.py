#!/usr/bin/python
# coding: utf-8

import logging
import tensorflow as tf
import pickle
import time
import numpy as np

from .active_nn import ActiveNeuralNetwork, compute_score
from .visualize_database import (plot_advancement_qdb_search,
                                 random_advancement_plot,
                                 plot_advancement_uncertainty_search,
                                 random_uncertainty_plot)
from .sampling import initial_sampling, uncertainty_sampling
from .qdbsampling import qdb_sampling
from threesetsmetric import threesetsmanager
from alex_library.tf_utils import utils


def active_search(X, y, shapes=[64, 1], max_iterations=501,
                  pos_weights_path="./weights_pos.pckl",
                  neg_weights_path="./weights_neg.pckl",
                  main_weights_path="./weights_main_%s_iterations.pckl",
                  display_plot=True, plot_save_path=None, n_points=None,
                  callback_save_path=None, nb_max_main_epoch=64000, qdb=True,
                  random=True, xlim=None, ylim=None, timer_save_path=None,
                  save_biased=True, include_background=False,
                  evolutive_small=False, nb_background_points=None,
                  nb_biased_epoch=10000, biased_lr=0.001, tsm=False,
                  tsm_lim=None, reduce_factor=None, pool_size=None,
                  main_lr=0.001, nn_activation="relu",
                  nn_loss="binary_crossentropy",
                  background_sampling="uncertain", loss_criteria=True):
    """
    Run the Query by disagreement search with neural networks.
    Params:
        X (np.array): Data on which the data exploration will be done.
        y (np.array): Searched labels.
        shapes (list of integers): shapes of the nn used.
        max_iterations (integer): maximum number of iterations.
        pos_weights_path (string): where to save the positively biased weights.
        neg_weights_path (string): where to save the negatively biased weights.
        main_weights_path (string): where to save the weights of the main nn.
        display_plot (boolean): Shall the script display the advancement plots?
        plot_save_path (string): Where to save the advancement plots. If None,
            the plots will not be saved.
        n_points (integer): Number of points to plot. If None, every points
            will be plotted.
        callback_save_path (string): Where to save the callbacks. If None,
            the callbacks will not be saved.
        qdb (boolean): If True, use query by disagreement to choose the new
            sample. Else, it uses uncertainty sampling.
        random (boolean): If True, take a random sample in the disagreement
            region. Else, take the most uncertain. Not applyable if not qdb.
        xlim (2-uple of integers): limits of the x axis.
        ylim (2-uple of integers): limits of the y axis.
        timer_save_path (string): where to save the follow up of the time of
            execution. If None, it will not be saved.
        save_biased (boolean): If True, will save weights of positive and
            negative nns.
        include_background (boolean): If True, the stopping criterion of biased
            nns will include bacground_points.
        evolutive_small (boolean): Choose if the number of background points
            will change over time or not.
        nb_biased_epoch (integer): Number of epoch to do at the first step.
        biased_lr (real): Learning rate of biased neural networks.
        tsm (boolean): Use or not three set
        tsm_lim (integerr): Maximum number of tuples examined during one
            iteration of tsm.update_regions
        reduce_factor (real or string): The gradients of the biased sample will
            be divided by this factor. If None, it will be equal to
            len(biased_sample). If "evolutive", it will be equal to
            len(biased_samples) * 2. / X_train.shape[0].
        pool_size (integer): Size of the pool considered to find the most
            uncertain point. If None, the whole X is used.
        main_lr (real): Learning rate of the main model.
        nn_activation (string): activation functions of the neural networks
        nn_loss (string): loss of the neural networks
        background_sampling (string). If "uncertain", background points will be
            the most uncertain of the model. If "random", background points
            will be randomly sampled.
    """

    # Initialize variables

    timer = {"sampling": [],
             "total": [],
             "main_nn": [],
             "iterations": [],
             "predictions": [],
             "saving_weights": [],
             "callback_treatment": [],
             "plots": [],
             "callback_save": [],
             "timer_save": []}

    if qdb:
        for key in ["background_points", "pos_nn", "neg_nn",
                    "disagreement_point"]:
            timer[key] = []

    t0 = time.time()
    graph_main = tf.Graph()
    input_shape = X.shape[1]

    with graph_main.as_default():
        nn_main = ActiveNeuralNetwork(
            input_shape=input_shape, hidden_shapes=shapes, batch_size=124,
            learning_rate=main_lr, activation=nn_activation, loss=nn_loss,
            )
    if qdb:
        graph_pos = tf.Graph()
        graph_neg = tf.Graph()

        with graph_pos.as_default():
            nn_pos = ActiveNeuralNetwork(
                input_shape=input_shape, hidden_shapes=shapes, batch_size=124,
                include_small=include_background, loss=nn_loss,
                learning_rate=biased_lr, activation=nn_activation,
                )
        with graph_neg.as_default():
            nn_neg = ActiveNeuralNetwork(
                input_shape=input_shape, hidden_shapes=shapes, batch_size=124,
                loss=nn_loss, include_small=include_background,
                learning_rate=biased_lr, activation=nn_activation,
                )

    with tf.Session(graph=graph_main) as sess_main:

        tf.set_random_seed(42)
        sess_main.run(tf.global_variables_initializer())
        nb_epoch_main = 1000

        stopping_criterion = False
        # Loop over the iterations
        for iteration in range(2, max_iterations):

            if stopping_criterion:
                break

            t = time.time()
            timer["total"].append(t - t0)

            logging.info("# Iteration %s #" % iteration)

            # Test if it is the first iteration
            if iteration == 2:

                # Randomly sample one positive and one negative example
                samples = initial_sampling(y)
                X_train, y_train = X[samples], y[samples]

                uncertain_samples = range(X.shape[0])
                for i in samples:
                    uncertain_samples.remove(i)
                X_val, y_val = X[uncertain_samples], y[uncertain_samples]

                tbis = time.time()
                timer["sampling"].append(tbis - t)
                t = tbis
                logging.info("Initial sampling done")

                # Train the neural network
                callback = nn_main.training(sess_main, X_train, y_train,
                                            n_epoch=100000, display_step=1000,
                                            stop_at_1=True, saving=False)

                callback["samples"] = samples
                # Initialize tsm
                if tsm:
                    tsm_object = threesetsmanager.ThreeSetsManager(
                        X_train, y_train, None, tsm_lim,
                        lower_bounds=np.min(X_val, axis=0),
                        upper_bounds=np.max(X_val, axis=0)
                        )

                tbis = time.time()
                timer["main_nn"].append(tbis - t)
                t = tbis

                # First prediction
                old_pred = sess_main.run(
                    nn_main.prediction,
                    feed_dict={nn_main.input_tensor: X_val}
                    )

                tbis = time.time()
                timer["predictions"].append(tbis - t)
                t = tbis

                val_score = compute_score(nn_main, X, y, sess_main)
                logging.info("Validation F1 score: %s" % val_score)

                callback["validation_error"] = [val_score]
                callback["training_error"] = [callback["training_error"][-1]]

                tbis = time.time()
                timer["callback_treatment"].append(tbis - t)
                timer["iterations"].append(t - timer["total"][-1] - t0)

            else:
                repeat = True
                while repeat:
                    repeat = False
                    if qdb:
                        sample, pred_pos, pred_neg, biased_samples, times = (
                            qdb_sampling(nn_main, sess_main, X_train, y_train,
                                         X_val, y_val, iteration, nn_pos,
                                         graph_pos, pos_weights_path, nn_neg,
                                         graph_neg, neg_weights_path,
                                         random=random, save=save_biased,
                                         evolutive_small=evolutive_small,
                                         nb_background_points=nb_background_points,
                                         nb_biased_epoch=nb_biased_epoch,
                                         reduce_factor=reduce_factor,
                                         pool_size=pool_size,
                                         loss_criteria=loss_criteria)
                            )

                        for i, key in enumerate([
                                "background_points", "pos_nn", "neg_nn",
                                "disagreement_point"
                                ]):
                            timer[key].append(times[i])
                    else:
                        sample = uncertainty_sampling(nn_main, sess_main,
                                                      X_val,
                                                      pool_size=pool_size)

                    if (tsm and
                            (tsm_object.get_label(X_val[sample]) is not None)):
                        X_train = np.vstack((X_train,
                                             X_val[sample].reshape((1, -1))))
                        y_train = np.vstack((
                            y_train, np.array([tsm_object.get_label(
                                               X_train[-1]
                                               )]).reshape((1, 1))
                            ))

                        if qdb:
                            if samples == 0:
                                biased_samples = [i - 1
                                                  for i in biased_samples]
                            elif sample != (X_val.shape[0] - 1):
                                biased_samples = [i - 1 if i > sample else i
                                                  for i
                                                  in biased_samples]

                            for variable in [X_val, y_val, pred_pos, pred_neg,
                                             old_pred]:
                                variable = np.delete(variable, sample, 0)
                        else:
                            for variable in [X_val, y_val]:
                                variable = np.delete(variable, sample, 0)

                        logging.info("I REPEAT !!!!!")
                        logging.info("Training main model with falsy new " +
                                     "sample !")
                        logging.info("Label of the new_sample: %s"
                                     % y_train[-1])
                        temp = nn_main.training(
                            sess_main, X_train, y_train, n_epoch=nb_epoch_main,
                            display_step=100000, saving=False,
                            stop_at_1=not loss_criteria, callback=True,
                            loss_criteria=loss_criteria
                            )

                        # Save the weights
                        if "%s" in main_weights_path:
                            utils.saver(
                                nn_main.params,
                                sess_main,
                                main_weights_path % iteration
                                )
                        else:
                            utils.saver(nn_main.params, sess_main,
                                        main_weights_path)

                        callback["training_error"].append(
                            temp["training_error"][-1]
                            )

                        # If the model did not converge, increase maximum
                        # training time next time.
                        if (callback["training_error"][-1] != 1):
                            if (2 * nb_epoch_main) >= nb_max_main_epoch:
                                nb_epoch_main = nb_max_main_epoch
                            else:
                                nb_epoch_main *= 2
                        repeat = True
                        val_score = compute_score(nn_main, X, y, sess_main)

                        logging.info("Validation F1 score: %s" % val_score)

                if tsm:
                    logging.info("Adding sample to tsm")
                    tsm_object.add_sample(X_val[sample], y_val[sample])
                    logging.info("Updating_new_regions")
                    tsm_object.update_regions()

                X_train = np.vstack((X_train,
                                     X_val[sample].reshape((1, -1))))
                y_train = np.vstack((y_train,
                                     y_val[sample].reshape((1, 1))))

                if qdb:
                    if samples == 0:
                        biased_samples = [i - 1 for i in biased_samples]
                    elif sample != (X_val.shape[0] - 1):
                        biased_samples = [i - 1 if i > sample else i
                                          for i
                                          in biased_samples]

                    for variable in [X_val, y_val, pred_pos, pred_neg,
                                     old_pred]:
                        variable = np.delete(variable, sample, 0)
                else:
                    for variable in [X_val, y_val]:
                        variable = np.delete(variable, sample, 0)

                tbis = time.time()
                timer["sampling"].append(tbis - t)
                t = tbis

                logging.info("New sample is %s" % bool(y_train[-1]))

                # Train the main nn with the new samples
                logging.info("Training main model")
                temp = nn_main.training(
                    sess_main, X_train, y_train, n_epoch=nb_epoch_main,
                    display_step=100000, saving=False, stop_at_1=not loss_criteria,
                    callback=True, loss_criteria=loss_criteria
                    )

                tbis = time.time()
                timer["main_nn"].append(tbis - t)
                t = tbis

                # Save the weights
                if "%s" in main_weights_path:
                    utils.saver(
                        nn_main.params,
                        sess_main,
                        main_weights_path % iteration
                        )
                else:
                    utils.saver(nn_main.params, sess_main, main_weights_path)

                tbis = time.time()
                timer["saving_weights"].append(tbis - t)
                t = tbis

                callback["training_error"].append(temp["training_error"][-1])

                # If the model did not converge, increase maximum training time
                # next time
                if (callback["training_error"][-1] != 1):
                    if (2 * nb_epoch_main) >= nb_max_main_epoch:
                        nb_epoch_main = nb_max_main_epoch
                    else:
                        nb_epoch_main *= 2

                tbis = time.time()
                timer["callback_treatment"].append(tbis - t)
                t = tbis

                val_score = compute_score(nn_main, X, y, sess_main)
                callback["validation_error"].append(val_score)

                logging.info("Validation F1 score: %s" % val_score)
                tbis = time.time()
                timer["predictions"].append(tbis - t)
                t = tbis

                stopping_criterion = (val_score > 0.99)

                if (display_plot or (plot_save_path is not None)):
                    # Predict with current model
                    new_pred = sess_main.run(
                        nn_main.prediction,
                        feed_dict={nn_main.input_tensor: X_val}
                        )

                    # Plot the progress
                    psp = plot_save_path
                    if (psp is not None) and ("%s" in psp):
                            psp = plot_save_path % iteration

                    if qdb:
                        if n_points is None:
                            plot_advancement_qdb_search(
                                X_train, y_train, X_val, y_val, old_pred,
                                new_pred, pred_pos, pred_neg, biased_samples,
                                save_path=psp, show=display_plot, xlim=xlim,
                                ylim=ylim
                                )
                        else:
                            random_advancement_plot(
                                X_train, y_train, X_val, y_val, old_pred,
                                new_pred, pred_pos, pred_neg, biased_samples,
                                n_points, save_path=psp, show=display_plot,
                                xlim=xlim, ylim=ylim
                                )
                    else:
                        if n_points is None:
                            plot_advancement_uncertainty_search(
                                X_train, y_train, X_val, y_val, old_pred,
                                new_pred, save_path=psp, show=display_plot,
                                xlim=xlim, ylim=ylim
                                )
                        else:
                            random_uncertainty_plot(
                                X_train, y_train, X_val, y_val, old_pred,
                                new_pred, n_points=n_points, save_path=psp,
                                show=display_plot, xlim=xlim, ylim=ylim)

                    old_pred = new_pred
                tbis = time.time()
                timer["plots"].append(tbis - t)
                t = tbis

                logging.info("Saving Callback")
                # Complete the callback
                callback["samples"].append(sample)

                # Save the callback
                if callback_save_path is not None:
                    with open(callback_save_path, "w") as fp:
                        pickle.dump(callback, fp)

                tbis = time.time()
                timer["callback_save"].append(tbis - t)
                t = tbis

                # Save the timer
                if timer_save_path is not None:
                    with open(timer_save_path, "w") as fp:
                        pickle.dump(timer, fp)

                tbis = time.time()
                timer["timer_save"].append(tbis - t)

                timer["iterations"].append(tbis - timer["total"][-1] - t0)

    logging.info("ENF OF STORY BRO")


if __name__ == "__main__":
    pass
