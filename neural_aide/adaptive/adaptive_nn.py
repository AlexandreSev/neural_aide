#!/usr/bin/python
# coding: utf-8

import logging
import numpy as np
import tensorflow as tf
import tf_utils as utils
from ..active_nn import ActiveNeuralNetwork as TrueActiveNN


def reduce_weights(x, k):
    """
    Divide x by 200 if x is not None
    """
    if x is not None:
        return x/k
    else:
        return None


def compute_score(nn, X, y, sess, f1score=True):
    """
    Compute a score of a neural network.
    Params:
        nn (ActiveNeuralNetwork): the nn to evaluate.
        X (np.array): the data on which the nn will be evaluated.
        y (np.array): Labels of X.
        f1score (boolean): If True, calculate f1score. Else, calculate
            accuracy.
    """
    pred = sess.run(nn.prediction, feed_dict={nn.input_tensor: X})
    if (len(y.shape) == 1) or (y.shape[1] == 1):
        if not f1score:
            return np.mean((pred > 0.5) == y)
        else:
            pred_yes = np.where(pred > 0.5)[0]
            if len(pred_yes) > 0:
                precision = np.mean(y[pred_yes] == 1)
            else:
                precision = 0
            true_yes = np.where(y == 1)[0]
            recall = np.mean(pred[true_yes] > 0.5)
            if ((precision+recall) == 0):
                return 0
            else:
                return 2. * precision * recall / (precision + recall)
    else:
        return np.mean(np.argmax(pred, 1) == np.argmax(y, 1))


class ActiveNeuralNetwork():
    """
    Create a feed forward network
    """

    def __init__(self, input_shape=2, hidden_shapes=[2, 1],
                 loss="binary_crossentropy", batch_size=64,
                 first_fully_connected_input_shape=3136, include_small=False,
                 learning_rate=0.001, activation="relu"):
        """
        Args:
            input_shape (int or tuple of int): Shape of the input of the
                neural network.
            first_hidden_shapes ([tuple of int or int]): list of the shape of the
                layer. If an int is given, it is a fully connected layer. If a
                tuple of the shape (a, b, c) is given, it is a convolutional
                layer with c filters of size a*a with stride b.
            loss (string): Either "l2", "binary_crossentropy" or
                "multiclass_crossentropy".
            batch_size (integer): Size of one batch during training
            include_small (boolean): If True, include X_small when computing
                the training score.
            learning_rate (real): learning rate used during the training.
        """

        self.input_shape = input_shape
        self.current_hidden_shapes = list(hidden_shapes)
        self.loss = loss
        self.batch_size = batch_size
        self.first_fully_connected_input_shape = first_fully_connected_input_shape
        self.include_small = include_small
        self.learning_rate = learning_rate

        self.nn = TrueActiveNN(input_shape, hidden_shapes,
                 loss, batch_size, first_fully_connected_input_shape,
                 include_small, learning_rate)

        self.update_attributes()

    def setLR(self, lr):
        self.nn.setLR(lr)

    def update_attributes(self):
        self.prediction = self.nn.prediction
        self.input_tensor = self.nn.input_tensor
        self.params = self.nn.params

    def training(self, sess, X_train, y_train, X_small=None, y_small=None,
                 X_val=None, y_val=None, n_epoch=100, callback=True,
                 saving=True, save_path="./model.ckpt", warmstart=False,
                 weights_path="./model.kpt", display_step=50, stop_at_1=False,
                 nb_min_epoch=50, reduce_factor=1, increase_if_not_1=False,
                 decrease=True):
        """
        Train the model on given data.

        Args:
            sess (tensorflow.Session): Current running session.
            X_train (np.array): Training data.
                Must have the shape (?, input_shape).
            y_train (np.array): Labels of X_train.
            X_small (np.array): Training data, with smaller weights.
            y_small (np.array): Labels of X_small.
            X_val (np.array): Data which determine the best model.
                Must have the shape (?, input_shape).
            y_val (np.array): Labels of X_val.
            n_epoch (int): Number of passes through the data.
            callback (boolean): If true, all errors will be saved in a
                dictionary.
            saving (boolean): If true, the best model will be saved in the
                save_dir folder.
            save_path (string): where to save the file if saving==True.
            warmstart (boolean): If true, the model will load weights from
                weights_path at the beginning.
            weights_path (string): Where to find the previous weights if
                warmstart=True.
            display_step (int): The number of epochs between two displays.
            stop_at_one (boolean): If true, training will be stopped when
                training score reach one.
            nb_min_epoch (int): Minimal number of epoch before the stop of the
                training
            reduce_factor (real): The gradients of X_small will be divided
                by this factor.

        Return:
            dictionnary: If callback==True, return the callback.
                Else, return an empty dictionnary
        """
        if decrease and self.current_hidden_shapes[0]>3:
            self.decrease_complexity(sess)

        total_X = X_train.copy()
        for X in [X_small, X_val]:
            if X is not None:
                total_X = np.vstack((total_X, X))

        self.test_dead_units(sess, total_X)

        callback = self.nn.training(sess, X_train, y_train, X_small, y_small,
                                    X_val, y_val, n_epoch, callback, saving,
                                    save_path, warmstart, weights_path,
                                    display_step, stop_at_1, nb_min_epoch,
                                    reduce_factor)

        if increase_if_not_1 and (callback["training_error"][-1] != 1):
            self.increase_complexity(sess)

            callback = self.nn.training(
                sess, X_train, y_train, X_small, y_small, X_val, y_val,
                10 * n_epoch, callback, saving, save_path, warmstart,
                weights_path, display_step, stop_at_1, nb_min_epoch,
                reduce_factor
                )

        return callback

    def decrease_complexity(self, sess, threshold=0.001):
        """
        Merge units that are very close.
        """

        toDelete = []

        dico_saver = utils.saver(self.nn.params, sess)
        
        current_W = np.vstack((
            dico_saver["weights"]["W0"],
            dico_saver["biases"]["b0"].reshape((1, -1))
            ))

        following_W = np.vstack((
            dico_saver["weights"]["W1"],
            ))

        for col_ind in range(current_W.shape[1]):
            if col_ind in toDelete:
                continue
            current_norm = np.linalg.norm(current_W[:, col_ind])
            for col_comp in range(col_ind+1, current_W.shape[1]):
                if col_comp in toDelete:
                    continue
                norm_ratio = (np.linalg.norm(current_W[:, col_ind] -
                                             current_W[:, col_comp]) 
                             / current_norm)
                if norm_ratio < threshold:

                    toDelete.append((col_comp, col_ind))
                    

        toDelete = sorted(toDelete, reverse=True)

        logging.debug("Here is toDelete: %s" %(toDelete))
        logging.debug("Here is the shape of following_W: %s, %s" %(following_W.shape))

        alreadyDeleted = []
        
        for (col_comp, col_ind) in toDelete:
            if col_comp in alreadyDeleted:
                continue
            alreadyDeleted.append(col_comp)
            following_W[col_ind, :] += following_W[col_comp, :]

            following_W = np.delete(following_W, col_comp, 0)

            current_W = np.delete(current_W, col_comp, 1)

        dico_saver["weights"]["W0"] = current_W[:-1,:]
        dico_saver["biases"]["b0"] = current_W[-1, :].reshape(-1)
        
        dico_saver["weights"]["W1"] = following_W

        if len(alreadyDeleted) > 0:
            self.current_hidden_shapes[0] -= len(alreadyDeleted)

            logging.info("Decreasing Complexity. New hidden shapes is %s"
                         % (self.current_hidden_shapes,))

            self.nn = TrueActiveNN(
                self.input_shape, self.current_hidden_shapes, self.loss,
                self.batch_size, self.first_fully_connected_input_shape,
                self.include_small, self.learning_rate)

            sess.run(tf.global_variables_initializer())
            self.update_attributes()

            utils.loaderFromDict(self.nn.params, sess, dico_saver)


    def increase_complexity(self, sess, loadPreviousWeights=True):
        """
        Increase the complexity of the neural network by doubling the number
        of units in the hidden layer.
        """
        old_shape = self.current_hidden_shapes[0]
        self.current_hidden_shapes[0] *= 2

        logging.info("Increasing Complexity. New hidden shapes is %s"
                     % (self.current_hidden_shapes,))

        # Load the previous weights
        if loadPreviousWeights:

            dico_saver = utils.saver(self.nn.params, sess)
            dico_saver_new = {"weights": {}, "biases": {}}

            #W0
            noise = np.random.normal(0, 0.001, dico_saver["weights"]["W0"].shape)
            dico_saver_new["weights"]["W0"] = np.hstack((
                dico_saver["weights"]["W0"] + noise,
                dico_saver["weights"]["W0"] - noise
                ))  / 2.
            #b0
            noise = np.random.normal(0, 0.001, dico_saver["biases"]["b0"].shape)
            dico_saver_new["biases"]["b0"] = np.concatenate((
                dico_saver["biases"]["b0"] + noise,
                dico_saver["biases"]["b0"] - noise
            )) / 2.

            #W1
            noise = np.random.normal(0, 0.001, dico_saver["weights"]["W1"].shape)
            dico_saver_new["weights"]["W1"] = np.vstack((
                dico_saver["weights"]["W1"] + noise,
                dico_saver["weights"]["W1"] - noise
                ))  / 2.
            dico_saver_new["biases"]["b1"] = dico_saver["biases"]["b1"]
            
        
        self.nn = TrueActiveNN(
            self.input_shape, self.current_hidden_shapes, self.loss,
            self.batch_size, self.first_fully_connected_input_shape,
            self.include_small, self.learning_rate)

        sess.run(tf.global_variables_initializer())
        self.update_attributes()

        if loadPreviousWeights:
            utils.loaderFromDict(self.nn.params, sess, dico_saver_new)

    def test_dead_units(self, sess, X):

        logging.debug("TEST DEAD UNITS")
        hidden_representations = sess.run(
            self.nn.hidden_representations[1],
            feed_dict={self.nn.input_tensor: X})

        hidden_sum = np.sum(hidden_representations, axis=0)

        logging.debug("Here is the shape of the hidden_sum: %s" % ([hidden_sum.shape]))

        toDelete = np.where(hidden_sum==0)[0]

        dico_saver = utils.saver(self.nn.params, sess)
        
        current_W = np.vstack((
            dico_saver["weights"]["W0"],
            dico_saver["biases"]["b0"].reshape((1, -1))
            ))

        following_W = dico_saver["weights"]["W1"]

        for col in sorted(toDelete, reverse=True):
            
            following_W = np.delete(following_W, col, 0)

            current_W = np.delete(current_W, col, 1)

        dico_saver["weights"]["W0"] = current_W[:-1,:]
        dico_saver["biases"]["b0"] = current_W[-1, :].reshape(-1)
        
        dico_saver["weights"]["W1"] = following_W

        if len(toDelete) > 0:
            self.current_hidden_shapes[0] -= len(toDelete)

            logging.info("Delete dead units. New hidden shapes is %s"
                         % (self.current_hidden_shapes,))

            self.nn = TrueActiveNN(
                self.input_shape, self.current_hidden_shapes, self.loss,
                self.batch_size, self.first_fully_connected_input_shape,
                self.include_small, self.learning_rate)

            sess.run(tf.global_variables_initializer())
            self.update_attributes()

            utils.loaderFromDict(self.nn.params, sess, dico_saver)



if __name__ == "__main__":
    logging.basicConfig(level=10)
    nn = ActiveNeuralNetwork()
