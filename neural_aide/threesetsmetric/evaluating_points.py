#!/usr/bin/python
# coding: utf-8

import numpy as np
import logging


def EvaluatingGridPoints(n_points, lower_bounds, upper_bounds):
    """
    A grid of point to evaluate the three sets metrics
    Params:
        n_points (integer): Number of points in the grid.
        lower_bounds (list of real): The minimum value for each dimension.
        upper_bounds (list of real): The macimum value for each dimension.

    Return:
        (np.array): An array of all the samples to create the grid.
    """

    # Find the dimension of the grid
    dim = len(lower_bounds)

    # Number of points per dimension
    n_points_per_dim = int(n_points ** (1./dim))

    # Space between two points in each dimension
    granularity = [(j - i) * 1. / (n_points_per_dim - 1)
                   for i, j in zip(lower_bounds, upper_bounds)]

    logging.debug("Here is granularity: %s" % (granularity,))

    # We create the grid in the lexicographic order.

    # First column of the grid
    grid = np.array(
        [lower_bounds[0] + j * granularity[0]
         for j in range(n_points_per_dim)] * (n_points_per_dim
                                              ** (dim - 1))
         ).reshape((-1, 1))

    # Create each new column and stack it to the grid.
    for i in range(1, dim):
        temp = np.array(
            list(
                np.repeat(
                    [lower_bounds[i] + j * granularity[i]
                     for j in range(n_points_per_dim)],
                    n_points_per_dim ** i)
                )
            * (n_points_per_dim ** (dim - i - 1))
            ).reshape((-1, 1))

        logging.debug("Here is grid shape: %s" % (grid.shape,))
        logging.debug("Here is temp shape: %s" % (temp.shape,))

        grid = np.hstack((grid, temp))

    return grid
