#!/usr/bin/python
# coding: utf-8

import logging
import numpy as np
import time
import tensorflow as tf

from .sampling import find_k_most_uncertain
from alex_library.tf_utils import utils


def training_biased_nn(X_train, y_train, X_val, y_val, nn, graph, weights_path,
                       biased_samples, positive=True, load_weights=True,
                       save=True, first_nb_epoch=10000, min_biased_epoch=100,
                       reduce_factor=None):
    """
    Train a positive neural network
    Params:
        X_train (np.array): Samples already labeled.
        y_train (np.array): Labels of X_train.
        X_val (np.array): Unlabeled samples.
        y_val (np.array): Labels of X_val.
        nn (ActiveNeuralNetwork): Biased nn to train.
        graph (tf.graph): Graph associated with the nn.
        weights_path (string): Where are saved and where to save the weights.
        biased_samples (list of integers): Indices of samples used to biase the
            model.
        positive (boolean): If True, the biased samples will be considered as
            positive samples, else as negative samples.
        load_weights (boolean): If True, will load previous wieghts.
        save (boolean): If True, will save weights.
        first_nb_epoch (integer): Number of epoch to do at the first step
        min_biased_epoch (integer): number minimal of epoch for the training
        reduce_factor (real or string): The gradients of the biased sample will
            be divided by this factor. If None, it will be equal to
            len(biased_sample). If "evolutive", it will be equal to
            len(biased_samples) * 2. / X_train.shape[0].
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
        repeat = False
        if reduce_factor is None:
            _reduce_factor = len(biased_samples)
        elif reduce_factor == "evolutive":
            _reduce_factor = len(biased_samples) * 2. / X_train.shape[0]
        else:
            _reduce_factor = reduce_factor
        # Loop while it does not predict a postive sample
        while criterion:

            # Train the neural network longer when it does not
            # predict a positive sample
            nb_epoch *= 2

            # Train the neural network
            if repeat:
                nn.training(sess, np.vstack((X_train, X_val[biased_samples])),
                            np.vstack((y_train, y_small)), n_epoch=nb_epoch,
                            display_step=100000, stop_at_1=True,
                            saving=False, callback=True,
                            nb_min_epoch=min_biased_epoch,
                            reduce_factor=_reduce_factor)
            else:
                nn.training(sess, X_train,
                            y_train, X_small=X_val[biased_samples],
                            y_small=y_small, n_epoch=nb_epoch,
                            display_step=100000, stop_at_1=True,
                            saving=False, callback=True,
                            nb_min_epoch=min_biased_epoch,
                            reduce_factor=_reduce_factor)

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
                 neg_weights_path, random=False, save=True,
                 evolutive_small=False, nb_biased_epoch=10000,
                 reduce_factor=None):
    """
    Find the next sample with query by disagreement.
    Params:
        nn_main (ActiveNeuralNetwork): Main nn of the active search.
        sess_main (tf.Session): sessions associated with the nn.
        X_train (np.array): Samples already labeled.
        y_train (np.array): Labels of X_train.
        X_val (np.array): Unlabeled samples.
        y_val (np.array): Labels of X_val.
        iteration (integer): number of the current iteration.
        nn_pos (ActiveNeuralNetwork): positively biased nn.
        graph_pos (tf.graph): graph associated to the nn_pos.
        pos_weight_path (string): where to save positive weights.
        nn_neg (ActiveNeuralNetwork): negatively biased nn.
        graph_neg (tf.graph): graph associated to the nn_neg.
        neg_weight_path (string): where to save negative weights.
        random (boolean): If True, take a random sample in the disagreement
            region. Else, take the most uncertain.
        save (boolean): If True, will save weights of postive and negative nns.
        evolutive_small (boolean): Choose if the number of background points
            will change over time or not.
        nb_biased_epoch (integer): Number of epoch to do at the first step
        reduce_factor (real or string): The gradients of the biased sample will
            be divided by this factor. If None, it will be equal to
            len(biased_sample). If "evolutive", it will be equal to
            len(biased_samples) * 2. / X_train.shape[0].

    Return:
        (integer) indice of the new sample
        (np.array) labels predicted by the postive model
        (np.array) labels predicted by the negative model
        (4-uple of float) time taken for each step
    """

    # Find background points
    t0 = time.time()
    if evolutive_small:
        biased_samples = find_k_most_uncertain(nn_main, sess_main, X_val,
                                               k=2*iteration)
    else:
        biased_samples = find_k_most_uncertain(nn_main, sess_main, X_val,
                                               k=200)

    t1 = time.time()
    # Training of biased nn.

    # Train the positively-biased network
    logging.info("Training positive model")
    pred_pos = training_biased_nn(
        X_train, y_train, X_val, y_val, nn_pos, graph_pos, pos_weights_path,
        biased_samples, True, (iteration != 3), save, nb_biased_epoch,
        reduce_factor=reduce_factor,
        )
    t2 = time.time()

    # Train the negatively-biased netswork
    logging.info("Training negative model")
    pred_neg = training_biased_nn(
        X_train, y_train, X_val, y_val, nn_neg, graph_neg, neg_weights_path,
        biased_samples, False, (iteration != 3), save, nb_biased_epoch,
        reduce_factor=reduce_factor,
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
