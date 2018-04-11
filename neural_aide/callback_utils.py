#!/usr/bin/python
# coding: utf-8

from matplotlib import pyplot as plt
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize'] = 16, 7


def treat_timer(timer):
    """
    Remove first and last iteration in order to make the callback consistent
    Params:
        timer (dict): timer callback return by active search

    Return:
        (dict): A dictionary with callbacks from the 3 to the penultimate
            iteration
    """
    for key in timer:

        if key not in ["timer_save", "callback_save", "saving_weights",
                       "plots", "background_points", "pos_nn", "neg_nn",
                       "disagreement_point"]:
            timer[key] = timer[key][1:]

        if key not in ["iterations", "timer_save"]:
            timer[key] = timer[key][:-1]
    return timer


def plot_sampling(timer, title=None):
    """
    Plot the timer for the sampling step
    """
    for key in ["iterations", "sampling", "background_points", "pos_nn",
                "neg_nn", "disagreement_point"]:
        if key in timer:
            plt.plot(timer[key], label=key)
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()


def plot_true_iterations(timer, title=None, show=True):
    """
    Plot the timer of the true computational time
        (without callbacks save, plots, and predictions).
    """
    to_plot = timer["iterations"]
    for key in ["timer_save", "predictions", "callback_treatment", "plots",
                "saving_weights", "callback_save"]:
        if key in timer:
            to_plot = [i - j for i, j in zip(to_plot, timer[key])]
    plt.plot(to_plot, label="iterations")
    plt.legend()
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def plot_true_breakdown(timer, title=None):
    """
    Plot the breakdown between nn training and sampling 
    """
    plot_true_iterations(timer, show=False)
    for key in ["main_nn", "sampling"]:
        plt.plot(timer[key], label=key)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import pickle

    path = "./timers/test.pckl"

    with open(path, "r") as fp:
        timer = pickle.load(fp)

    timer = treat_timer(timer)
    for key in timer:
        if key != "total":
            plt.plot(timer[key], label=key)
    plt.legend()
    plt.show()
