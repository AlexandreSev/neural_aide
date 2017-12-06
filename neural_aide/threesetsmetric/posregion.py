#!/usr/bin/python
# coding: utf-8

import numpy as np

from .facet import Facet


class PosRegion():
    """
    Implement the convex polytope
    """

    def __init__(self, pos_samples):
        """
        Params:
            pos_samples (np.array): dim+1 positive samples to create the
                (dim)-polytope.
        """
        self.dim = pos_samples.shape[1]

        if (self.dim+1) != pos_samples.shape[0]:
            raise ValueError("Wrong number of samples")

        self.vertices = pos_samples
        self.facets = []

        self.create_facets()

    def create_facets(self):
        """
        Create the facets of the polytope
        """
        # For each sample in the set of vertices, create the facet that does
        # not contain this sample
        for sample_id in range(self.vertices.shape[0]):
            facet_points = np.delete(self.vertices, (sample_id), axis=0)
            self.facets.append(
                Facet(facet_points, self.vertices[sample_id, :])
                )

    def contain(self, point):
        """
        Check if a point is inside the positive region.
        A point is inside the positive region if it is not visible by any of
        the facets of the positive region.
        Params:
            point (np.array): point to check.

        Return:
            (boolean): True if point is inside the positive region.
        """
        contain = True
        for facet in self.facets:
            if facet.is_visible(point):
                contain = False
                break
        return contain

    def add_vertex(self, point):
        """
        Add a new vertex on the positive region.
        Params:
            point (np.array): point to add to the positive region.
        """

        # Step 1: Find visible facets
        visible_facets = []
        for facet in self.facets:
            if facet.is_visible(point):
                visible_facets.append(facet)

        # If there is no visible facets, the point is inside the positive
        # region, do don't do anything.
        if not visible_facets:
            return None

        # Step 2: find ridges that connect a visible facet and a hidden facet.
        # They are also the ridges that only occurs once in the set of visible
        # facets.
        horizon_ridges = []
        hash_horizon_ridges = []  # Use hash to skip arrays comparison

        horizon_ridges = []
        # Work first with hash to skip array comparing issues
        hash_horizon_ridges = []

        for facet in visible_facets:

            self.facets.remove(facet)

            for ridge in facet.get_ridges():

                if hash(ridge.tostring()) in hash_horizon_ridges:
                    hash_horizon_ridges.remove(hash(ridge.tostring()))
                else:
                    hash_horizon_ridges.append(hash(ridge.tostring()))

        # Finally, use ridge
        for facet in visible_facets:
            for ridge in facet.get_ridges():
                if hash(ridge.tostring()) in hash_horizon_ridges:
                    horizon_ridges.append(ridge)

        # Step 3: Add facets with the new points and horizon ridges
        for ridge in horizon_ridges:

            for point_id in range(self.vertices.shape[0]):
                if self.vertices[point_id, :] not in ridge:
                    ref = self.vertices[point_id, :]
                    break

            self.facets.append(
                Facet(np.vstack((ridge, point)), ref)
                )

        # Finally, update the vertices of this region
        self.vertices = np.vstack((
                self.vertices,
                point.reshape((1, -1)),
                ))
        self.clean_vertices()

    def clean_vertices(self):
        """
        Remove vertices that are not on a facet
        """
        to_remove = []
        for vertex_id in range(self.vertices.shape[0]):
            current_vertex = self.vertices[vertex_id, :]

            is_useful = False
            for facet in self.facets:
                if current_vertex in facet.vertices:
                    is_useful = True

            if not is_useful:
                to_remove.append(vertex_id)

        self.vertices = np.delete(self.vertices, to_remove, 0)
