#!/usr/bin/python
# coding: utf-8
import logging
import numpy as np
from . import negregion
from . import posregion

class ThreeSetsManager():
    """
    Implement the whole three set metric stuff
    """

    def __init__(self, X_train, y_train, X_val, lim=None):
        """
        """
        # All the positive samples
        self.pos_samples = X_train[y_train.reshape(-1) == 1, :]
        # All the negative samples
        self.neg_samples = X_train[y_train.reshape(-1) == 0, :]
        # Negative sample used to build a negtive region
        self.neg_useful_samples = []
        # All the uncertain samples
        self.uncertain_samples = X_val
        # Handle samples
        self.remaining_samples = np.arange(X_val.shape[0])

        self.dim = X_train.shape[1]
        
        self.pos_region = None
        self.neg_regions = None

        if lim is None:
            self.lim = X_val.shape[0]
        else:
            self.lim = lim
        self.counter = 0

    def add_sample(self, sample, label):
        """
        Add a sample, and update the three regions
        Params:
            sample (integer): Id of the sample in the uncertain region
            label (integer): 0 or 1. Label of the sample
        """

        # Add the sample to the corresponding region
        new_sample = self.uncertain_samples[sample]

        if label == 0:
            self.neg_samples = np.vstack((self.neg_samples, 
                new_sample.reshape((1, self.dim))))
        else:
            self.pos_samples = np.vstack((self.pos_samples, 
                new_sample.reshape((1, self.dim))))

        # Update the uncertain region
        self.uncertain_samples = np.delete(self.uncertain_samples, sample, 0)
        self.remaining_samples = np.delete(self.remaining_samples, sample, 0)
            
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
                    self.add_neg_point(self, new_sample)

            # If we have dim + 1 samples, create the positive region
            elif self.pos_samples.shape[0] == (self.dim + 1):

                self.pos_region = posregion.PosRegion(self.pos_samples)
        
        else:
            if label == 0:
                self.add_neg_point(new_sample)
            else:
                if not self.pos_region.contain(new_sample):
                    self.pos_region.add_vertex(new_sample)
                    for neg_region in self.neg_regions:
                        neg_region.add_vertex(new_sample)

    
    def add_neg_point(self, new_sample):
        """
        """ 
        new_sample_useful = True
                
        # check if the new sample bring some information
        for neg_region in self.neg_regions:
            if neg_region.contain(new_sample):
                new_sample_useful = False
                break
        # Add the new sample if it is useful, and check if another
        # negative region became irelevant with this new one.
        if new_sample_useful:
            if self.pos_region is not None:
                self.neg_regions.append(negregion.NegRegion(new_sample, None,
                    self.pos_region))
            else:
                self.neg_regions.append(negregion.NegRegion(new_sample, 
                    self.pos_samples))
            self.neg_useful_samples.append(self.neg_samples.shape[0] - 1)

            index = 0
            while index < (len(self.neg_useful_samples) - 1):
                sample_id = self.neg_useful_samples[index]
                if self.neg_regions[-1].contain(self.neg_samples[sample_id]):
                    self.neg_useful_samples.remove(sample_id)
                    self.neg_regions.remove(self.neg_regions[index])
                else:
                    index += 1

    def update_regions(self):
        """
        """
        to_remove = []
        for sample_id in range(self.counter, self.counter + self.lim):
            sample = self.uncertain_samples[sample_id, :]
            if ((self.pos_region is not None) and 
                    (self.pos_region.contain(sample))):
                to_remove.append(sample_id)
                self.pos_samples = np.vstack((self.pos_samples, 
                                              sample.reshape((1, -1))))
            elif self.neg_regions is not None:
                for neg_region in self.neg_regions:
                    if neg_region.contain(sample):
                        to_remove.append(sample_id)
                        self.neg_samples = np.vstack((self.neg_samples, 
                                              sample.reshape((1, -1))))

        self.uncertain_samples = np.delete(
            self.uncertain_samples, to_remove, 0
            )
        self.remaining_samples = np.delete(
            self.remaining_samples, to_remove, 0
            )
        self.counter = ((self.counter + self.lim) % 
                         self.uncertain_samples.shape[0])
        self.lim = np.minimum(self.lim, self.uncertain_samples.shape[0])

    def get_new_sets(self):
        """
        """
        X_train = np.vstack((self.pos_samples, self.neg_samples))
        y_train = np.vstack((
            np.ones((self.pos_samples.shape[0], 1)),
            np.zeros((self.neg_samples.shape[0], 1))
            ))

        order = np.arange(X_train.shape[0])
        np.random.shuffle(order)

        return (X_train[order], y_train[order], self.remaining_samples)
