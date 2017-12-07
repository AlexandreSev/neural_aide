#!/usr/bin/python
# coding: utf-8

import numpy as np


def f1_score(y, pred):
    """
    Compute the f1score
    Params:
        y (np.array): True labels.
        pred (np.array): Prediceted labels.

    Return:
        (real): f1score
    """
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


def create_naive_score(callback, labels):
    """
    Read the callback to create the f1score obtained by predicting correctly
    the training example and the same labels for the others.
    Params:
        callback (dict): callback saved by active search
        labels (np.array): true labels of data

    Return:
        (list of real): naive f1score at each iterations.
    """

    samples = callback["samples"]
    answer = []
    for index in range(2, len(samples)+1):
        pred_pos = np.ones((labels.shape[0], 1))
        pred_neg = np.zeros((labels.shape[0], 1))

        pred_pos[samples[:index]] = labels[samples[:index]]
        pred_neg[samples[:index]] = labels[samples[:index]]
        answer.append(max(f1_score(labels, pred_pos),
                          f1_score(labels, pred_neg)))

    return answer


if __name__ == "__main__":
    import pickle
    import json

    with open("/Users/alex/Documents/LIX-PHD/experiments/clean_active_nn/test.pckl", "r") as fp:
        callback = pickle.load(fp)

    with open("/Users/alex/Documents/LIX-PHD/experiments/clean_active_nn/ressources/housing_dataset/alex_labels.json", "r") as fp:
        labels = np.array(json.load(fp)).reshape((-1, 1))

    print(callback["samples"])
    print(create_naive_score(callback, labels))
