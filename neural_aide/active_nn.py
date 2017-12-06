#!/usr/bin/python
# coding: utf-8

import logging
import numpy as np
import tensorflow as tf
from alex_library.tf_utils import utils


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


class ActiveNeuralNetwork:
    """
    Create a feed forward network
    """

    def __init__(self, input_shape=2, hidden_shapes=[64, 1],
                 loss="binary_crossentropy", batch_size=64,
                 first_fully_connected_input_shape=3136, include_small=False,
                 learning_rate=0.001):
        """
        Args:
            input_shape (int or tuple of int): Shape of the input of the
                neural network.
            hidden_shapes ([tuple of int or int]): list of the shape of the
                layer. If an int is given, it is a fully connected layer. If a
                tuple of the shape (a, b, c) is given, it is a convolutional
                layer with c filters of size a*a with a stride b.
            loss (string): Either "l2", "binary_crossentropy" or
                "multiclass_crossentropy".
            batch_size (integer): Size of one batch during training
            include_small (boolean): If True, include X_small when computing
                the training score.
            learning_rate (real): learning rate used during the training.
        """

        self.sizes = [input_shape] + hidden_shapes

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        if loss in ["l2", "binary_crossentropy", "multiclass_crossentropy"]:
            self.loss_name = loss
        else:
            raise ValueError("Wrong Value for parameter loss")

        self.batch_size = batch_size

        self.unknown_shape = first_fully_connected_input_shape
        self.include_small = include_small

        self.build()

    def build(self):
        """
        Lead the construction of the tensorflow graph by calling appropriate
            functions.
        """
        self.create_weights()
        self.create_placeholder()
        self.build_model()

    def create_weights(self):
        """
        Create the weights of the model
        """
        logging.debug("ENTERING create_weights")
        self.params = {"weights": {}, "biases": {}}
        for i in range(1, len(self.sizes)):

            logging.debug("i = %s" % i)
            logging.debug("sizes[i] = " + str(self.sizes[i]))

            if type(self.sizes[i]) == int:

                if type(self.sizes[i-1]) == int:

                    self.params["biases"]["b%s" % (i-1)] = (
                        utils.create_bias_variable(shape=[self.sizes[i]])
                    )

                    self.params["weights"]["W%s" % (i-1)] = (
                        utils.create_weight_variable(shape=(self.sizes[i-1],
                                                            self.sizes[i]))
                    )
                else:

                    self.params["biases"]["b%s" % (i-1)] = (
                        utils.create_bias_variable(shape=[self.sizes[i]])
                    )

                    self.params["weights"]["W%s" % (i-1)] = (
                        utils.create_weight_variable(shape=(self.unknown_shape,
                                                            self.sizes[i]))
                    )
            else:

                self.params["biases"]["b%s" % (i-1)] = (
                    utils.create_bias_variable(shape=[self.sizes[i][2]])
                )

                self.params["weights"]["W%s" % (i-1)] = (
                    utils.create_weight_variable(shape=(self.sizes[i][0],
                                                        self.sizes[i][0],
                                                        self.sizes[i-1][2],
                                                        self.sizes[i][2]))
                )
        logging.debug("END OF create_weights")

    def create_placeholder(self):
        """
        Create the placeholders of the model.
        """
        if type(self.sizes[0]) == int:
            self.input_tensor = tf.placeholder(dtype=tf.float32,
                                               shape=(None, self.sizes[0]))
        else:
            self.input_tensor = tf.placeholder(
                dtype=tf.float32,
                shape=(None, self.sizes[0][0],
                       self.sizes[0][1], self.sizes[0][2])
                )
        self.true_labels = tf.placeholder(dtype=tf.float32,
                                          shape=(None, self.sizes[-1]))
        self.reduce_factor = tf.placeholder(dtype=tf.float32)

    def build_model(self):
        """
        Create the operation between weights and placeholders
        """

        self.hidden_representations = [self.input_tensor]
        current_input = self.input_tensor

        for i in range(len(self.sizes)-2):

            if type(self.sizes[i+1]) == int:
                # current_input = tf.contrib.layers.flatten(current_input)

                current_input = tf.nn.relu(
                    tf.matmul(
                        current_input,
                        self.params["weights"]["W%s" % i]
                    ) + self.params["biases"]["b%s" % i]
                )

                self.hidden_representations.append(current_input)

            else:
                current_input = tf.nn.relu(
                    tf.nn.conv2d(
                        current_input,
                        self.params["weights"]["W%s" % i],
                        (1, self.sizes[i + 1][1], (self.sizes[i + 1][1], 1),
                         "SAME")
                    ) + self.params["biases"]["b%s" % i]
                )

                current_input = tf.nn.max_pool(current_input, (1, 2, 2, 1),
                                               (1, 2, 2, 1), "VALID")
                self.hidden_representations.append(current_input)

        # current_input = tf.contrib.layers.flatten(current_input)

        if self.loss_name == "l2":
            self.prediction = tf.sigmoid(tf.matmul(
                current_input,
                self.params["weights"]["W%s" % (len(self.sizes)-2)]
                ) + self.params["biases"]["b%s" % (len(self.sizes)-2)])

            self.loss = tf.nn.l2_loss(self.true_labels - self.prediction)

        else:
            self.out_tensor = tf.matmul(
                current_input,
                self.params["weights"]["W%s" % (len(self.sizes)-2)]
                ) + self.params["biases"]["b%s" % (len(self.sizes)-2)]

            if self.loss_name == "binary_crossentropy":

                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.true_labels,
                    logits=self.out_tensor
                    )
                self.prediction = tf.sigmoid(self.out_tensor)

            elif self.loss_name == "multiclass_crossentropy":

                self.prediction = tf.nn.softmax(self.out_tensor)
                self.loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.true_labels,
                    logits=self.out_tensor
                    )

        self.training_step = self.optimizer.minimize(self.loss)

        compute_gradients = self.optimizer.compute_gradients(self.loss)
        small_gradients = [(reduce_weights(i, self.reduce_factor), j)
                           for i, j in compute_gradients]

        self.apply_small_gradients = self.optimizer.apply_gradients(
            small_gradients
            )

    def training(self, sess, X_train, y_train, X_small=None, y_small=None,
                 X_val=None, y_val=None, n_epoch=100, callback=True,
                 saving=True, save_path="./model.ckpt", warmstart=False,
                 weights_path="./model.kpt", display_step=50, stop_at_1=False,
                 nb_min_epoch=50, reduce_factor=1):
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

        if warmstart:
            utils.loader(self.params, sess, weights_path)

        if (X_val is not None) & (y_val is not None):
            best_val_error = compute_accuracy(self, X_val, y_val, sess)
            best_n_epoch = 0
        else:
            best_val_error = None
            best_n_epoch = None

        dico_callback = {"training_error": [], "validation_error": [],
                         "testing_error": []}

        # Loop over epochs
        for nb_epoch in range(1, n_epoch + 1):

            # Run the training step
            dico_callback, best_val_error, best_n_epoch = (
                self.training_one_step(nb_epoch, sess, X_train, y_train,
                                       X_small, y_small, X_val, y_val,
                                       best_val_error, best_n_epoch, callback,
                                       dico_callback, display_step, saving,
                                       save_path, reduce_factor)
                )

            if stop_at_1 and (nb_epoch >= nb_min_epoch):
                if dico_callback["training_error"][-1] == 1:
                    break

        self.display(nb_epoch, sess, X_train, y_train, X_small, y_small, X_val,
                     y_val, best_val_error, best_n_epoch)

        if callback:
            return dico_callback
        else:
            return {}

    def training_one_step(self, nb_epoch, sess, X_train, y_train, X_small=None,
                          y_small=None, X_val=None, y_val=None,
                          best_val_error=None, best_n_epoch=None,
                          callback=True, dico_callback={}, display_step=50,
                          saving=True, save_path="./model.ckpt",
                          reduce_factor=1):
        """
        Run the training for 1 epoch.

        Args:
            nb_epoch (int): Number of the current epoch.
            sess (tensorflow.Session): Current running session.
            X_train (np.array): Training data.
                Must have the shape (?, input_shape)
            y_train (np.array): Labels of X_train.
            X_small (np.array): Training data, with smaller weights.
            y_small (np.array): Labels of X_small.
            X_val (np.array): Data which determine the best model.
                Must have the shape (?, input_shape)
            y_val (np.array): Labels of X_val.
            best_val_error (float): Minimum error encountered yet.
            best_n_epoch (int): On which epoch best_val_error has been found.
            callback (boolean): If true, all errors will be saved in a
                dictionary
            dico_callback (dictionary): dictionary where callbacks are saved,
                if callback is True.
            display_step (int): The number of epochs between two displays.
            saving (boolean): If true, the best model will be saved in the
                save_dir folder.
            save_path (string): where to save the file if saving==True.
            reduce_factor (real): the gradients of X_small will be divided by
                this factor

        Returns:
            dictionary: dictionary where callbacks are saved,
                if callback is True.
            float: Minimum error encountered yet.
            int: On which epoch the mininimum error has been found.
        """

        feed_dict = {}

        i = -1
        for i in range(X_train.shape[0]/self.batch_size):
            feed_dict_temp = feed_dict
            feed_dict_temp[self.input_tensor] = X_train[
                i * self.batch_size: (i+1) * self.batch_size
                ]
            feed_dict_temp[self.true_labels] = y_train[i * self.batch_size:
                                                       (i+1) * self.batch_size]

            sess.run(self.training_step, feed_dict=feed_dict_temp)

        feed_dict_temp = feed_dict
        feed_dict_temp[self.input_tensor] = X_train[(i+1) * self.batch_size:]
        feed_dict_temp[self.true_labels] = y_train[(i+1) * self.batch_size:]

        sess.run(self.training_step, feed_dict=feed_dict_temp)

        if (X_small is not None) and (y_small is not None):
            feed_dict_temp = feed_dict
            feed_dict_temp[self.input_tensor] = X_small
            feed_dict_temp[self.true_labels] = y_small
            feed_dict_temp[self.reduce_factor] = reduce_factor

            sess.run(self.apply_small_gradients, feed_dict=feed_dict_temp)

        if (X_val is not None) & (y_val is not None):
            val_error = compute_accuracy(self, X_val, y_val, sess)
            if val_error > best_val_error:
                if saving:
                    utils.saver(self.params, sess, save_path)
                best_val_error = val_error
                best_n_epoch = nb_epoch

        if callback:
            if self.include_small and (X_small is not None):
                dico_callback["training_error"].append(compute_score(
                    self,
                    np.vstack((X_train, X_small)),
                    np.vstack((y_train, y_small)),
                    sess
                ))
            else:
                dico_callback["training_error"].append(compute_score(
                    self, X_train, y_train, sess
                ))

            if (X_val is not None) & (y_val is not None):
                dico_callback["validation_error"].append(val_error)

        if nb_epoch % display_step == 0:
            self.display(nb_epoch, sess, X_train, y_train, X_small, y_small,
                         X_val, y_val, best_val_error, best_n_epoch)

        return (dico_callback, best_val_error, best_n_epoch)

    def display(self, nb_epoch, sess, X_train, y_train, X_small=None,
                y_small=None, X_val=None, y_val=None, best_val_error=None,
                best_n_epoch=None):
        """
        Display some usefull information

        Args:
            nb_epoch (int): number of current epoch.
            sess (tensorflow.Session): current running session.
            X_train (np.array): Training data.
            y_train (np.array): Labels of X_train.
            X_small (np.array): Training data, with smaller weights.
            y_small (np.array): Labels of X_small.
            X_val (np.array): Data which determine the best model.
            y_val (np.array): Labels of X_val.
            best_val_error (float): Minimum error encountered yet.
            best_n_epoch (int): On which epoch best_val_error has been found.
        """
        logging.info("Epoch %s" % nb_epoch)
        if self.include_small and (X_small is not None):
            logging.info("Train Score %s" % compute_score(
                self,
                np.vstack((X_train, X_small)),
                np.vstack((y_train, y_small)), sess),
            )
        else:
            logging.info("Train Score %s" % compute_score(self,
                                                          X_train, y_train,
                                                          sess))

        if (X_val is not None) & (y_val is not None):
            logging.info("Validation Score %s" % compute_score(self, X_val,
                                                               y_val, sess))
            logging.info("Best model: %s epochs with Score %s" % (
                best_n_epoch, best_val_error
            ))


if __name__ == "__main__":
    logging.basicConfig(level=10)
    nn = ActiveNeuralNetwork()
