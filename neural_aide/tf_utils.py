# coding: utf-8
import tensorflow as tf
import numpy as np
import pickle
import os
import logging
import argparse


def create_weight_variable(shape, name="W"):
    """
    Create a tensorflow variable, initiale with a normal law.

    Args:
        shape (list or tupl): shape of the variable (None for undefined dimension)
        name (str): Name of the variable in the tensorflow graph

    Returns:
        tf.Variable: the tensorflow variable created
    """

    return tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(3. / (shape[0] + shape[1]))), name=name)
          
def create_bias_variable(shape, name="b"):
    """
    Create a tensorflow variable, initiale with a constant.

    Args:
        shape (list or tupl): shape of the variable (None for undefined dimension)
        name (str): Name of the variable in the tensorflow graph

    Returns:
        tf.Variable: the tensorflow variable created
    """
    return tf.Variable(tf.constant(0.0, shape=shape), name=name)

def saver(params, sess, save_path=None):
    """
    Custom saver for tensorflow model

    Args:
        params (dict): dictionnaire containing "weights" and "biases", in which are stocked the tensorflow 
                variables corresponding to the weights and the biases of the model.
        sess (tf.Session): Current tensorflow Session
        save_path (str): The path where the model has to be saved

    Return:
        dico_saver
    """
    weights = params["weights"]
    biases = params["biases"]
    weights_saver = {key: sess.run(weights[key]) for key in weights}
    biases_saver = {key: sess.run(biases[key]) for key in biases}
    dico_saver = {"weights": weights_saver, "biases": biases_saver}
    if save_path is not None:
        with open(save_path, "w") as fp:
            pickle.dump(dico_saver, fp)

    return dico_saver

def loaderFromDict(params, sess, dico_saver):
    """
    Custom loader for tensorflow model. Careful, the model must be created,
    it will only assign values to existing variable.

    Args:
        params (dict):  dictionnaire containing "weights" and "biases", in which are stocked the tensorflow 
                variables corresponding to the weights and the biases of the model.
        sess (tf.Session): Current tensorflow Session
        dico_saver(dict): the dictionnary with parameters will come.
    """
    weights = params["weights"]
    biases = params["biases"]
    weights_saver = dico_saver["weights"]
    biases_saver = dico_saver["biases"]
    for key in weights:
        if key in weights_saver:
            sess.run(weights[key].assign(weights_saver[key]))
        else:
            print("WARNING: key %s in params but not in saved weights.")
    for key in biases:
        if key in biases_saver:
            sess.run(biases[key].assign(biases_saver[key]))
        else:
            print("WARNING: key %s in params but not in saved biases.")

def loader(params, sess, save_path):
    """
    Custom loader for tensorflow model. Careful, the model must be created, it will only assign values
    to existing variable.

    Args:
        params (dict):  dictionnaire containing "weights" and "biases", in which are stocked the tensorflow 
                variables corresponding to the weights and the biases of the model.
        sess (tf.Session): Current tensorflow Session
        save_path (str): The path where the model is saved
    """
    with open(save_path, "r") as fp:
        dico_saver = pickle.load(fp)
    loaderFromDict(params, sess, dico_saver)
    

def loader_from_another_network(params, sess, params_saver):
    """
    Custom loader for tensorflow model. Careful, the model must be created, it will only assign values
    to existing variable.

    Args:
        params (dict):  dictionnaire containing "weights" and "biases", in which are stocked the tensorflow 
                variables corresponding to the weights and the biases of the model.
        sess (tf.Session): Current tensorflow Session
        save_path (str): The path where the model is saved
    """
    weights = params["weights"]
    biases = params["biases"]
    weights_saver = params_saver["weights"]
    biases_saver = params_saver["biases"]
    for key in weights:
        if key in weights_saver:
            sess.run(weights[key].assign(sess.run(weights_saver[key])))
        else:
            print("WARNING: key %s in params but not in saved weights.")
    for key in biases:
        if key in biases_saver:
            sess.run(biases[key].assign(sess.run(biases_saver[key])))
        else:
            print("WARNING: key %s in params but not in saved biases.")

def compute_accuracy(nn, X, y, sess, dropout=False, ae_ffnn=False, multiclass=False):
    """
    Compute the accuracy for a binary classification

    Args:
        nn (model): model which must have out_tensor, input_tensor and dropout attribute
        X (np.array): data on which the accuracy is computed
        y (np.array): labels of the data
        sess (tf.Session): current running Session
        dropout (float): keeping probability for the input. Set to None if there is no dropout.

    Returns:
        float: The computed accuracy
    """
    feed_dict = {nn.input_tensor: X}
    if dropout:
        for key in nn.dropout:
            feed_dict[key] = 1.
    if ae_ffnn:
        return np.mean((sess.run(nn.out_tensor_nn, feed_dict=feed_dict)>0.5) == y)
    elif multiclass:
        return np.mean(np.argmax(sess.run(nn.out_tensor, feed_dict=feed_dict), axis=-1) == np.argmax(y, axis=-1))
    else:
        return np.mean((sess.run(nn.out_tensor, feed_dict=feed_dict)>0.5) == y)

def compute_reconstruction_error(ae, X, sess, y=None, dropout=False):
    """
    Compute the reconstruction error for a autoencoder.

    Args:
        ae (model): auto_encoder which have loss, input_tensor and can have dropout attribute.
        X (np.array): data on wich the error is calculted
        sess (tf.Session): current running Session
        dropout (Boolean): Is the autoencoder denoising ?

    Returns:
        float: the computed error
    """
    if dropout:
        feed_dict = {ae.input_tensor: X}
        for i in ae.dropout:
            feed_dict[i] = 1.
    else:
        feed_dict = {ae.input_tensor: X}
    if y is not None:
        feed_dict[ae.true_labels] = y.reshape((-1,))
    return sess.run(ae.loss, feed_dict=feed_dict)

def logging(nb_buckets, hidden_sizes, denoising_rate):
    print("_"*30)
    print("")
    print('{:^30}'.format("  KFOLD %s"%nb_buckets))
    print("_"*30)
    print("{:^30}".format("Hidden_sizes: %s"%hidden_sizes))
    print("{:^30}".format("Denoising rate: %s"%denoising_rate))
    print("_"*30)
    print("")


def create_dico_mean(path, n_buckets=5):
    """
    Average the values of a dictionary on the number of buckets
    Parameters:
        path: path of the initial dictionary
        n_buckets: number of buckets in the cross validation
    """
    with open(path, "r") as fp:
        dico_callback = pickle.load(fp)
        
    dico_mean = dico_callback["bucket_0"]
    for n in range(1, n_buckets):
        for key in dico_mean:
            dico_mean[key] = [i + j for i, j in zip(dico_mean[key], dico_callback["bucket_%s"%n][key])]
    
    for key in dico_mean:
        dico_mean[key] = [i/n_buckets for i in dico_mean[key]]

    return dico_mean


def initialize_logger(log_file=None, filemode="a"):
    """
    Initialize a logger with a command line argument --log
    Params:
        log_file (string): Where to save the log.
            If None, logs will be printed on console.
    """
    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='Logging level', default="info")

    # Read the arguments
    args = parser.parse_args()

    # Verify each argument
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)

    # Configure logger
    logging.basicConfig(level=numeric_level, filename=log_file, 
        filemode=filemode, format='%(asctime)s %(levelname)s: %(message)s')
