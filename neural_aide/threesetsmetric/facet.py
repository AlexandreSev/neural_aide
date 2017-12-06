#!/usr/bin/python
# coding: utf-8

import numpy as np
import logging


class Facet():
    """
    Implement a facet of a convex polytope
    """

    def __init__(self, points, ref, refisneg=True):
        """
        Params:
            points (np.array): vertices of the facet
            ref (np.array): Used to orientate the facet. The point ref will be
                negative.
            refisneg (boolean): If True, the ref has to be negative. Else, it
                has to be positive.
        """
        self.vertices = points
        self.refisneg = refisneg

        # Sort the vertices
        order = np.argsort(self.vertices[:, 0])
        self.vertices = self.vertices[order, :]

        self.dim = points.shape[1]

        self.create_coef()

        self.check_ref(ref)

    def __repr__(self):
        return "Facet of vertices %s" % (self.vertices,)

    def create_coef(self):
        """
        Create the coefficients of the hyperplane describe by the facet. The
        orientation is futher defined with check_ref.
        """
        if np.linalg.matrix_rank(self.vertices) == self.vertices.shape[1]:
            logging.debug("Here are my vertices: %s" % (self.vertices,))
            self.coefs = np.linalg.solve(self.vertices, np.ones((self.dim, 1)))
            norm = np.linalg.norm(self.coefs)
            self.coefs = self.coefs.T / norm
            self.offset = -1. / norm
        else:
            raise ValueError("The matrix of points is not of full rank")

    def check_ref(self, ref):
        """
        Oriente the facet such that ref is negative
        Params:
            ref (np.array): point which have to be negative
        """
        if ((self.is_visible(ref) and self.refisneg)
                or (not self.is_visible(ref) and not self.refisneg)):
            self.coefs *= -1
            self.offset *= -1

    def is_visible(self, point):
        """
        Check if a point is on the positive side of the facet
        Params:
            point (np.array): point to check

        Return:
            (boolean): True if the point is visible
        """
        return (self.coefs.dot(point.reshape((-1, 1))) + self.offset > 0)

    def get_ridges(self):
        """
        Get the ridge of the facet. It is composed of dim (dim-2)-polytopes.
        Return:
            (list of np.array): every ridges describes py their points
        """
        ridges = []
        for i in range(self.vertices.shape[0]):
            ridges.append(np.delete(self.vertices, i, 0))
        return ridges
