#!/usr/bin/python
# coding: utf-8

import logging
import tensorflow as tf
import pickle
import time
import numpy as np

from .active_nn import ActiveNeuralNetwork, compute_score
from .visualize_database import (plot_advancement_qdb_search,
                                 random_advancement_plot)
from threesetsmetric import threesetsmanager
from alex_library.tf_utils import utils


def find_k_most_uncertain(nn, sess, X, batch_size=128, k=2,
                          multiclass=False):
    """
    Find the most uncertain sample for a neural network in a database.
    Params:
        nn (ActiveNeuralNetwork): The neural network evaluated.
        sess (tf.Session): The session associated to the nn.
        X (np.array): The database used.
        batch_size (integer): size of the batch during evaluation.
        k (integer): number of samples to be returned.
        multiclass (boolean): Consider the prediction of the neural
            network to be a softmax if True, else a sigmoid.

    Return:
        (list of integers): k most uncertain indices
    """

    results = []
    i = -1

    # Loop over the batches
    for i in range(X.shape[0]/batch_size):

        feed_dict = {}
        feed_dict[nn.input_tensor] = X[i * batch_size: (i+1) * batch_size]

        pred = sess.run(nn.prediction, feed_dict=feed_dict)

        if not multiclass:
            pred = np.sort(np.abs(0.5 - pred), axis=-1)
        else:
            pred = np.sort(np.max(pred, axis=-1)).reshape((-1, 1))
        results = kmin(results, zip(pred, range(i * batch_size,
                       i * batch_size + pred.shape[0])), k)

    # Last uncomplete batch
    feed_dict = {}
    feed_dict[nn.input_tensor] = X[(i+1) * batch_size:]

    pred = sess.run(nn.prediction, feed_dict=feed_dict)

    if not multiclass:
        pred = np.sort(np.abs(0.5 - pred), axis=-1)
    else:
        pred = np.sort(np.max(pred, axis=-1)).reshape((-1, 1))

    results = kmin(results, zip(pred, range((i+1) * batch_size,
                   (i + 1) * batch_size + pred.shape[0])), k)

    return [i[1] for i in results]


def kmin(l1, l2, k):
    """
    Return the k smaller elements of two lists. If there are not enough
        elements, return all of them, sorted.
    Params:
        l1 (list): first list. Must be sorted.
        l2 (list): second list. Must be sorted.
        k (integer): number of elements to return.

    Return:
        (list): k smaller elements
    """
    results = []
    while len(results) < k:
        if l1 == []:
            if l2 == []:
                return results
            else:
                results.append(l2[0])
                l2 = l2[1:]
        else:
            if l2 == []:
                results.append(l1[0])
                l1 = l1[1:]
            else:
                if l1[0][0] > l2[0][0]:
                    results.append(l2[0])
                    l2 = l2[1:]
                else:
                    results.append(l1[0])
                    l1 = l1[1:]
    return results


def initial_sampling(y):
    """
    Pick randomly two samples which have a different label
    Params:
        y (np.array): labels

    Return:
        (list of integers): indices of picked samples
    """
    samples = list(np.random.randint(0, len(y), 2))
    while len(np.unique(y[samples] > 0.5)) != 2:
        samples = list(np.random.randint(0, len(y), 2))
    return samples


def training_biased_nn(X_train, y_train, X_val, y_val, nn, graph, weights_path,
                       biased_samples, positive=True, load_weights=True,
                       save=True, first_nb_epoch=10000, min_biased_epoch=100):
    """
    Train a positive neural network
    Params:
        X (np.array): Database used in the active search.
        y (np.array): Labels of X.
        nn (ActiveNeuralNetwork): Biased nn to train.
        graph (tf.graph): Graph associated with the nn.
        weights_path (string): Where are saved and where to save the weights.
        samples (list of integers): List of indices of labelled samples.
        biased_samples (list of integers): Indices of samples used to biase the
            model.
        positive (boolean): If True, the biased samples will be considered as
            positive samples, else as negative samples.
        load_weights (boolean): If True, will load previous wieghts.
        save (boolean): If True, will save weights.
        first_nb_epoch (integer): Number of epoch to do at the first step
        min_biased_epoch (integer): number minimal of epoch for the training
    """
    with tf.Session(graph=graph) as sess:

        nb_epoch = first_nb_epoch/2

        # Initialize weights and load previous one if they exist
        sess.run(tf.global_variables_initializer())
        if load_weights:
            utils.loader(nn.params, sess, weights_path)

        if positive:
            y_small = np.ones((len(biased_samples), 1))
        else:
            y_small = np.zeros((len(biased_samples), 1))

        criterion = True

        # Loop while it does not predict a postive sample
        while criterion:

            # Train the neural network longer when it does not
            # predict a positive sample
            nb_epoch *= 2

            # Train the neural network
            nn.training(sess, X_train,
                        y_train, X_small=X_val[biased_samples],
                        y_small=y_small, n_epoch=nb_epoch,
                        display_step=100000, stop_at_1=True,
                        saving=False, callback=True,
                        nb_min_epoch=min_biased_epoch)

            # Predict the whole database
            pred = sess.run(nn.prediction,
                            feed_dict={nn.input_tensor: X_val})

            if positive:
                criterion = (np.sum(pred > 0.5) == 0)
            else:
                criterion = (np.sum(pred <= 0.5) == 0)
        # Save the weights for the next iteration
        if save:
            utils.saver(nn.params, sess, weights_path)
    return pred


def qdb_sampling(nn_main, sess_main, X_train, y_train, X_val, y_val, iteration,
                 nn_pos, graph_pos, pos_weights_path, nn_neg, graph_neg,
                 neg_weights_path, multiclass=False, random=False, save=True,
                 evolutive_small=False, nb_biased_epoch=10000):
    """
    Find the next sample with query by disagreement.
    Params:
        nn_main (ActiveNeuralNetwork): Main nn of the active search.
        sess_main (tf.Session): sessions associated with the nn.
        X (np.array): Database used during the active search.
        y (np.array): True labels.
        iteration (integer): number of the current iteration.
        samples (list of integer): indices of labeled samples.
        nn_pos (ActiveNeuralNetwork): positively biased nn.
        graph_pos (tf.graph): graph associated to the nn_pos.
        pos_weight_path (string): where to save positive weights.
        nn_neg (ActiveNeuralNetwork): negatively biased nn.
        graph_neg (tf.graph): graph associated to the nn_neg.
        neg_weight_path (string): where to save negative weights.
        multiclass (boolean): If true, consider the output of the nns to be a
            softmax, else a sigmoid.
        random (boolean): If True, take a random sample in the disagreement
            region. Else, take the most uncertain.
        save (boolean): If True, will save weights of postive and negative nns.
         evolutive_small (boolean): Choose if the number of background points
            will change over time or not.
        nb_biased_epoch (integer): Number of epoch to do at the first step

    Return:
        (integer) indice of the new sample
        (np.array) labels predicted by the postive model
        (np.array) labels predicted by the negative model
        (4-uple of float) time taken for each step
    """
    # Find 200 background point for the training of biases nns.
    t0 = time.time()
    if evolutive_small:
        biased_samples = find_k_most_uncertain(nn_main, sess_main, X_val,
                                               k=2*iteration,
                                               multiclass=multiclass)
    else:
        biased_samples = find_k_most_uncertain(nn_main, sess_main, X_val,
                                               k=200, multiclass=multiclass)

    t1 = time.time()
    # Training of biased nn.
    # Loop while it does not find a good sample.

    # Train the positively-biased network
    logging.info("Training positive model")
    pred_pos = training_biased_nn(
        X_train, y_train, X_val, y_val, nn_pos, graph_pos, pos_weights_path,
        biased_samples, True, (iteration != 3), save, nb_biased_epoch
        )
    t2 = time.time()

    # Train the negatively-biased netswork
    logging.info("Training negative model")
    pred_neg = training_biased_nn(
        X_train, y_train, X_val, y_val, nn_neg, graph_neg, neg_weights_path,
        biased_samples, False, (iteration != 3), save, nb_biased_epoch
        )
    t3 = time.time()

    # Look at the samples differently predicted by the biased network
    differences = ((pred_pos > 0.5) != (pred_neg > 0.5))
    if random:
        order = np.where(differences)[0]
        np.random.shuffle(order)
    else:
        order = np.where(differences)[0]
        temp = np.abs(
            np.abs(pred_pos[order] - np.mean(pred_pos[order]))
            + np.abs(pred_neg[order] - np.mean(pred_neg[order]))
            ).reshape(-1)
        order = order[np.argsort(temp)].reshape(-1)

    logging.info("Number of differences between positive " +
                 "and negative models: %s" % np.sum(differences))

    # Find a sample in differences which is not already present
    if len(order) > 0:
        logging.debug("Here is order: " + str(order))
        t4 = time.time()
        timers = (t1-t0, t2-t1, t3-t2, t4-t3)
        return order[0], pred_pos, pred_neg, biased_samples, timers

    # If the two models agree, choose the first biased point
    t4 = time.time()
    timers = (t1-t0, t2-t1, t3-t2, t4-t3)
    return biased_samples[0], pred_pos, pred_neg, biased_samples, timers

def uncertainty_sampling(nn_main, sess_main, X, multiclass=False):
    """
    Find the next sample with query by disagreement.
    Params:
        nn_main (ActiveNeuralNetwork): Main nn of the active search.
        sess_main (tf.Session): sessions associated with the nn.
        X (np.array): Database used during the active search.
        multiclass (boolean): If true, consider the output of the nns to be a
            softmax, else a sigmoid.

    Return:
        (integer) indice of the new sample
    """

    temp = find_k_most_uncertain(nn_main, sess_main, X, k=1,
                                 multiclass=multiclass)

    return temp[0]


def active_search(X, y, shapes=[64, 1], max_iterations=501,
                  pos_weights_path="./weights_pos.pckl",
                  neg_weights_path="./weights_neg.pckl",
                  main_weights_path="./weights_main_%s_iterations.pckl",
                  display_plot=True, plot_save_path=None, n_points=None,
                  callback_save_path=None, nb_max_main_epoch=64000, qdb=True,
                  random=True, xlim=None, ylim=None, timer_save_path=None,
                  save_biased=True, include_background=False,
                  evolutive_small=False, nb_biased_epoch=10000,
                  biased_lr=0.001, tsm=False, tsm_lim=None):
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
            loss="binary_crossentropy")
    if qdb:
        graph_pos = tf.Graph()
        graph_neg = tf.Graph()

        with graph_pos.as_default():
            nn_pos = ActiveNeuralNetwork(
                input_shape=input_shape, hidden_shapes=shapes, batch_size=124,
                loss="binary_crossentropy", include_small=include_background,
                learning_rate=biased_lr
                )
        with graph_neg.as_default():
            nn_neg = ActiveNeuralNetwork(
                input_shape=input_shape, hidden_shapes=shapes, batch_size=124,
                loss="binary_crossentropy", include_small=include_background,
                learning_rate=biased_lr
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
                                            n_epoch=100000, display_step=100,
                                            stop_at_1=True, saving=False)

                # Initialize tsm
                if tsm:
                    tsm_object = threesetsmanager.ThreeSetsManager(
                        X_train, y_train, X_val, tsm_lim
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

                if qdb:
                    sample, pred_pos, pred_neg, biased_samples, times = (
                        qdb_sampling(nn_main, sess_main, X_train, y_train,
                                     X_val, y_val, iteration, nn_pos,
                                     graph_pos, pos_weights_path, nn_neg,
                                     graph_neg, neg_weights_path,
                                     random=random, save=save_biased,
                                     evolutive_small=evolutive_small,
                                     nb_biased_epoch=nb_biased_epoch)
                        )

                    for i, key in enumerate([
                            "background_points", "pos_nn", "neg_nn",
                            "disagreement_point"
                            ]):
                        timer[key].append(times[i])
                else:
                    sample = uncertainty_sampling(nn_main, sess_main, X_val)

                if tsm:
                    logging.info("Adding sample to tsm")
                    tsm_object.add_sample(sample, y_val[sample])
                    logging.info("Updating_new_regions")
                    tsm_object.update_regions()
                    logging.info("Getting new sets")
                    X_train, y_train, remaining_samples = (
                        tsm_object.get_new_sets()
                            )
                    logging.info("New training size: %s" % X_train.shape[0])
                else:
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
                    if not tsm:
                        for variable in [X_val, y_val, pred_pos, pred_neg,
                                         old_pred]:
                            variable = np.delete(variable, sample, 0)
                    else:
                        for variable in [X_val, y_val, pred_pos, pred_neg,
                                         old_pred]:
                            variable = variable[remaining_samples, :]
                elif not tsm:
                    for variable in [X_val, y_val]:
                        variable = np.delete(variable, sample, 0)
                else:
                    for variable in [X_val, y_val]:
                        variable = variable[remaining_samples, :]

                tbis = time.time()
                timer["sampling"].append(tbis - t)
                t = tbis

                logging.info("New sample is %s" % bool(y_train[-1]))

                # Train the main nn with the new samples
                logging.info("Training main model")
                temp = nn_main.training(
                    sess_main, X_train, y_train, n_epoch=nb_epoch_main,
                    display_step=100000, saving=False, stop_at_1=True,
                    callback=True
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

                if (display_plot or (plot_save_path is not None)) and qdb:
                    # Predict with current model
                    new_pred = sess_main.run(
                        nn_main.prediction,
                        feed_dict={nn_main.input_tensor: X_val}
                        )

                    # Plot the progress
                    psp = plot_save_path
                    if (psp is not None) and ("%s" in psp):
                            psp = plot_save_path % iteration

                    if n_points is None:
                        plot_advancement_qdb_search(
                            X_train, y_train, X_val, y_val, old_pred, new_pred,
                            pred_pos, pred_neg, biased_samples, save_path=psp,
                            show=display_plot, xlim=xlim, ylim=ylim
                            )
                    else:
                        random_advancement_plot(
                            X_train, y_train, X_val, y_val, old_pred, new_pred,
                            pred_pos, pred_neg, biased_samples, n_points,
                            save_path=psp, show=display_plot, xlim=xlim,
                            ylim=ylim
                            )

                    old_pred = new_pred
                tbis = time.time()
                timer["plots"].append(tbis - t)
                t = tbis

                logging.info("Saving Callback")
                # Complete the callback
                callback["samples"] = samples

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
