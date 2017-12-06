#!/usr/bin/python
# coding: utf-8
import logging
import numpy as np
from . import negregion
from . import posregion
from . import evaluating_points


class ThreeSetsManager():
    """
    Implement the whole three set metric stuff
    """

    def __init__(self, X_train, y_train, X_val=None, lim=None, n_points=5000,
                 lower_bounds=None, upper_bounds=None):
        """
        X_train (np.array): two first points labeled
        y_train (np.array): labels associated to X_train
        X_val (np.array): Points on which the tsm will be evaluated.
            If None, a grid will be created.
        lim (integer): Maximum number of considered sample during one update
            of regions.
        n_points (integer): Number of points to create if X_val is None.
        lower_bounds (list of integers): Lower bounds of the grid.
        upper_bounds (list of integers): Upper bounds of the grid.
        """

        # All the positive samples
        self.pos_samples = X_train[y_train.reshape(-1) == 1, :]

        # All the negative samples
        self.neg_samples = X_train[y_train.reshape(-1) == 0, :]

        # Negative sample used to build a negtive region
        self.neg_useful_samples = []

        # All the uncertain samples
        if X_val is None:
            self.uncertain_samples = evaluating_points.EvaluatingGridPoints(
                n_points, lower_bounds, upper_bounds
                )
        else:
            self.uncertain_samples = X_val

        # Handle samples
        self.remaining_samples = np.arange(self.uncertain_samples.shape[0])

        # Dimension we are working in
        self.dim = X_train.shape[1]

        # Initialization of the positive and the negative region
        self.pos_region = None
        self.neg_regions = None

        # Put a upper limit on the number of samples treated per
        # update_regions call.
        if lim is None:
            self.lim = self.uncertain_samples.shape[0]
        else:
            self.lim = lim
        self.counter = 0

    def add_sample(self, sample, label):
        """
        Add a sample, and update the three regions
        Params:
            sample (integer or np.array): Id of the sample in the uncertain
                region or new sample.
            label (integer): 0 or 1. Label of the sample
        """

        # Create new_sample, the coordinates of the new sample
        if type(sample) == int:
            new_sample = self.uncertain_samples[sample]
        else:
            new_sample = sample

        # Add the sample to the corresponding set of samples
        if label == 0:
            self.neg_samples = np.vstack((
                self.neg_samples,
                new_sample.reshape((1, self.dim))
                ))
        else:
            self.pos_samples = np.vstack((
                self.pos_samples,
                new_sample.reshape((1, self.dim))
                ))

        # Update the uncertain region
        if type(sample) == int:
            self.uncertain_samples = np.delete(self.uncertain_samples, sample,
                                               0)
            self.remaining_samples = np.delete(self.remaining_samples, sample,
                                               0)

        # Update the positive and negative region

        if self.pos_region is None:

            # If the positive region has not be initialised
            if (self.pos_samples.shape[0] == self.dim):

                # If we have enough positive points to create the negative
                # region, either we just added a new positive sample and we
                # initialize the negative region with all these points, either
                # we already initialised the negative region, and we added a
                # new negative sample.

                if (self.neg_regions is None):
                    self.neg_regions = []
                    for sample_id in range(self.neg_samples.shape[0]):
                        self.neg_regions.append(
                            negregion.NegRegion(
                                self.neg_samples[sample_id],
                                self.pos_samples
                                )
                            )

                        logging.debug("Here is neg_region: %s" %
                                      self.neg_regions[-1])

                        self.neg_useful_samples.append(sample_id)
                else:
                    self.add_neg_point(new_sample)

            # If we have dim + 1 samples, create the positive region
            elif self.pos_samples.shape[0] == (self.dim + 1):

                self.pos_region = posregion.PosRegion(self.pos_samples)

        else:
            # If self.pos_region is not None, the negative region is also not
            # None
            if label == 0:
                self.add_neg_point(new_sample)

            else:
                if not self.pos_region.contain(new_sample):
                    self.pos_region.add_vertex(new_sample)
                    for neg_region in self.neg_regions:
                        neg_region.add_vertex(new_sample)

    def add_neg_point(self, new_sample):
        """
        Add a negative point when the positive region contains at least dim
        points.
        Params:
            new_sample (np.array): coordinates of the new negative sample.
        """
        new_sample_useful = True

        # check if the new sample brings some information
        for neg_region in self.neg_regions:
            if neg_region.contain(new_sample):
                new_sample_useful = False
                break

        # Add the new sample if it is useful, and check if another
        # negative region became irrelevant with this new one.
        if new_sample_useful:

            # Create new negative region
            if self.pos_region is not None:
                self.neg_regions.append(negregion.NegRegion(new_sample, None,
                                                            self.pos_region))
            else:
                self.neg_regions.append(negregion.NegRegion(new_sample,
                                                            self.pos_samples))

            # Store the new negative sample as a relevant sample.
            self.neg_useful_samples.append(self.neg_samples.shape[0] - 1)

            # Check if some relevant negatives samples became irrelevant.
            index = 0
            while index < (len(self.neg_useful_samples) - 1):

                sample_id = self.neg_useful_samples[index]

                # A sample is irrelevant if it is contained in the new negative
                # region. We can so remove the corresponding negative region.
                if self.neg_regions[-1].contain(self.neg_samples[sample_id]):
                    self.neg_useful_samples.remove(sample_id)
                    self.neg_regions.remove(self.neg_regions[index])

                else:
                    index += 1

    def update_regions(self):
        """
        Goes through uncertain samples to see if some can be labeled via the
        positive/negative regions.
        """

        to_remove = []
        # For each sample:
        for sample_id in range(self.counter, self.counter + self.lim):
            # Trick to limit the number of samples updates in one call.
            true_sample_id = sample_id % self.uncertain_samples.shape[0]
            sample = self.uncertain_samples[true_sample_id, :]

            # Check the positive region
            if ((self.pos_region is not None) and
                    (self.pos_region.contain(sample))):
                to_remove.append(true_sample_id)
                self.pos_samples = np.vstack((self.pos_samples,
                                              sample.reshape((1, -1))))
            # Check the negatives regions
            elif self.neg_regions is not None:
                for neg_region in self.neg_regions:
                    if neg_region.contain(sample):
                        to_remove.append(true_sample_id)
                        self.neg_samples = np.vstack((self.neg_samples,
                                                      sample.reshape((1, -1))))

        # Remove the labeled samples
        self.uncertain_samples = np.delete(
            self.uncertain_samples, to_remove, 0
            )
        self.remaining_samples = np.delete(
            self.remaining_samples, to_remove, 0
            )

        # Update the counter and the limit.
        self.counter = ((self.counter + self.lim) %
                        self.uncertain_samples.shape[0])
        self.lim = np.minimum(self.lim, self.uncertain_samples.shape[0])

    def get_new_sets(self):
        """
        Get the positive samples, the negative samples and the uncertain
        samples.

        Return:
            (np.array, np.array, np.array): Shuffled labeled set, associated
                labels and the id of the remaining uncertain samples.
        """
        X_train = np.vstack((self.pos_samples, self.neg_samples))
        y_train = np.vstack((
            np.ones((self.pos_samples.shape[0], 1)),
            np.zeros((self.neg_samples.shape[0], 1))
            ))

        order = np.arange(X_train.shape[0])
        np.random.shuffle(order)

        return (X_train[order], y_train[order], self.remaining_samples)

    def get_label(self, sample):
        """
        Check if we can label a sample with the positive/negative regions.
        Params:
            sample (np.array): The sample to check.

        Return:
            (integer): 1 if it is in the positive region, 0 if it is in a
                negative region, else None.
        """
        if (self.pos_region is not None) and (self.pos_region.contain(sample)):
            return(1)
        if self.neg_regions is not None:
            for neg_region in self.neg_regions:
                if neg_region.contain(sample):
                    return(0)
        return None
