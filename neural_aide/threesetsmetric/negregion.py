#!/usr/bin/python
# coding: utf-8

import numpy as np
import logging

from .facet import Facet


class NegRegion():
    """
    Implement the complement of a convex polytope to create negative region
    for active search
    """

    def __init__(self, neg_sample, pos_samples, pos_region=None):
        """
        Params:
            neg_sample (np.array): negative point, corner of the negative area
                to create.
            pos_samples (np.array): dim positive samples to create the negative
                region.
            pos_region (PosRegion): positive region to create the negative
                region associated. Used only if pos_samples is None.
        """
        self.neg_sample = neg_sample.reshape((1, -1))
        self.dim = self.neg_sample.shape[1]

        # Creation with dim points and not a positive region
        if (pos_samples is not None):

            if ((self.dim) != pos_samples.shape[0]):
                raise ValueError("Wrong number of samples")

            self.pos_vertices = pos_samples
            self._create_from_points()

        else:
            # Creation with a region
            if pos_region is None:
                raise Value("pos_samples or pos_region has to be different " +
                            "from None")

            self._create_from_region(pos_region)

    def __repr__(self):
        to_print = "Negative Region with facets "
        to_print += "; ".join([i.__repr__() for i in self.facets])
        to_print += " and useless facets "
        to_print += "; ".join([i.__repr__() for i in self.useless_facets])
        return to_print

    def _create_from_points(self):
        """
        First way to create a region: 1 negative sample et dim positive samples
        """
        self.facets = []

        self._create_facets()

    def _create_facets(self):
        """
        Create the facets of the polytope
        """
        # For each sample in the set of vertices, create the facet that does
        # not contain this sample
        for sample_id in range(self.pos_vertices.shape[0]):

            logging.debug("sample id: %s" % sample_id)
            logging.debug("neg_sample shape: %s" % (self.neg_sample.shape,))
            logging.debug("pos_vertice shape: %s" % (self.pos_vertices.shape,))

            facet_points = np.vstack((
                self.neg_sample,
                np.delete(self.pos_vertices, (sample_id), axis=0),
                ))

            self.facets.append(
                Facet(facet_points, self.pos_vertices[sample_id, :],
                      refisneg=False)
                )
        # Create facet which does not contain the negative point
        self.useless_facets = [Facet(self.pos_vertices, self.neg_sample,
                                     refisneg=True)]

    def _create_from_region(self, region):
        """
        Second way to create a negative region: 1 negative sample and a
            positive region
        Params:
            region (PosRegion): positive region used.
        """

        # Step 1: find visible facets of the positive region

        visible_facets = []
        for facet in region.facets:
            if facet.is_visible(self.neg_sample):
                visible_facets.append(facet)

        logging.debug("Visible facets: %s" % (visible_facets,))

        # Step 2: find ridges that connect a visible facet and a hidden facet.
        # They are also the ridges that only occurs once in the set of visible
        # facets.

        horizon_ridges = []
        # Work first with hash to skip array comparing issues
        hash_horizon_ridges = []

        for facet in visible_facets:

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

        logging.debug("Horizon ridges %s" % (horizon_ridges,))
        # Step 3: creating negative region with points on the horizon ridges.
        self.pos_vertices = np.unique(np.vstack((horizon_ridges)), axis=0)

        logging.debug("Pos vertices in create_from_region: ")
        logging.debug(str(self.pos_vertices))

        if self.pos_vertices.shape[0] == self.dim:
            self._create_from_points()
        else:
            # If we have more than dim positive vertices, create the area from
            # dim points and add the others.
            self.pos_vertices_to_add = self.pos_vertices[self.dim:]

            self.pos_vertices = self.pos_vertices[:self.dim]
            self._create_from_points()

            for sample_id in range(self.pos_vertices_to_add.shape[0]):
                self.add_vertex(self.pos_vertices_to_add[sample_id])

    def contain(self, point):
        """
        Check if a point is inside the negative region.
        A point is inside the negative region if it is not visible by any of
        the facets of the negative region.
        Params:
            point (np.array): point to check.

        Return:
            (boolean): True if point is inside the negative region.
        """
        contain = True
        for facet in self.facets:
            if facet.is_visible(point):
                contain = False
                break
        return contain

    def add_vertex(self, point):
        """
        Add a new positive vertex on the negative region.
        Params:
            point (np.array): point to add.
        """

        # Step 1: Find invisible facets
        invisible_facets = []
        for facet in self.facets:
            if not facet.is_visible(point):
                invisible_facets.append(facet)

        logging.debug(" POWPOWPOW Invisible_facets: %s" % (invisible_facets,))

        # If there is no visible facets or no invisible facets, the point does
        # not bring new information
        if ((not invisible_facets) or
                (len(invisible_facets) == len(self.facets))):
            return None

        # Add facet which does not include the negative sample:
        for facet in self.useless_facets:
            if not facet.is_visible(point):
                invisible_facets.append(facet)

        logging.debug(" POWPOWPOW 2 Invisible_facets: %s"
                      % (invisible_facets,))

        # Step 2: find ridges that connect a visible facet and a hidden facet.
        # They are also the ridges that only occurs once in the set of
        # invisible facets.

        horizon_ridges = []

        # Work first with hash to skip array comparing issues
        hash_horizon_ridges = []

        for facet in invisible_facets:

            if facet in self.facets:
                self.facets.remove(facet)
            else:
                self.useless_facets.remove(facet)

            for ridge in facet.get_ridges():

                if hash(ridge.tostring()) in hash_horizon_ridges:
                    hash_horizon_ridges.remove(hash(ridge.tostring()))
                else:
                    hash_horizon_ridges.append(hash(ridge.tostring()))

        # Finally, use ridge
        for facet in invisible_facets:
            for ridge in facet.get_ridges():
                if hash(ridge.tostring()) in hash_horizon_ridges:
                    horizon_ridges.append(ridge)

        logging.debug("POWPOWPOW horizon_ridges %s" % (horizon_ridges,))

        # Step 3: Add facets with the new points and horizon ridges
        for ridge in horizon_ridges:

            for point_id in range(self.pos_vertices.shape[0]):
                if self.pos_vertices[point_id, :] not in ridge:
                    ref = self.pos_vertices[point_id, :]

                    break

            if self.neg_sample in ridge:
                self.facets.append(
                    Facet(np.vstack((ridge, point)), ref, refisneg=False)
                    )
            else:
                self.useless_facets.append(
                    Facet(np.vstack((ridge, point)), ref, refisneg=True)
                    )

        # Finally, update the vertices of this region
        self.pos_vertices = np.vstack((
                self.pos_vertices,
                point.reshape((1, -1)),
                ))

        self.clean_vertices()

    def clean_vertices(self):
        """
        Remove vertices that are not on a facet
        """
        to_remove = []
        for vertex_id in range(self.pos_vertices.shape[0]):
            current_vertex = self.pos_vertices[vertex_id, :]

            is_useful = False
            for facet in self.facets + self.useless_facets:
                if current_vertex in facet.vertices:
                    is_useful = True

            if not is_useful:
                to_remove.append(vertex_id)

        self.pos_vertices = np.delete(self.pos_vertices, to_remove, 0)
