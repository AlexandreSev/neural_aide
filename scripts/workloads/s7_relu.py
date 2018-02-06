#!/usr/bin/python
# coding: utf-8

from neural_aide import main
import logging
import traceback
import sys
from os.path import join as pjoin


RESSOURCES_FOLDER = "/Users/alex/Documents/LIX-PHD/experiments/clean_active_nn/"
QDB = True
RANDOM = True
SHAPES = [16, 16, 1]
INCLUDE_BACKGROUND = True
EVOLUTIVE_SMALL = True
NB_BIASED_EPOCH = 2000
USE_MAIN_WEIGHTS = False
DISPLAY = False
SAVE_PLOT = False
QUERY = 1
BIASED_LR = 0.001
TSM = False
PLTLIM = None
TSM_LIM = None
REDUCE_FACTOR = None
POOL_SIZE = None
MAIN_LR = 0.001
ACTIVATION = "relu"
LOSS = "binary_crossentropy"
BACKGROUND_SAMPLING = "uncertain"

for i in range(10):
    try:
        main.run_experiment_with_sdss(
            RESSOURCES_FOLDER,
            qdb=QDB,
            random=RANDOM,
            shapes=SHAPES,
            include_background=INCLUDE_BACKGROUND,
            evolutive_small=EVOLUTIVE_SMALL,
            nb_biased_epoch=NB_BIASED_EPOCH,
            use_main_weights=USE_MAIN_WEIGHTS,
            display=DISPLAY,
            save_plot=SAVE_PLOT,
            query=QUERY,
            biased_lr=BIASED_LR,
            tsm=TSM,
            pltlim=PLTLIM,
            tsm_lim=TSM_LIM,
            reduce_factor=REDUCE_FACTOR,
            pool_size=POOL_SIZE,
            np_seed=i+100,
            tf_seed=i,
            saving_dir=pjoin(RESSOURCES_FOLDER, "results", "s7_relu"),
            main_lr=MAIN_LR,
            nn_activation=ACTIVATION,
            nn_loss=LOSS,
            background_sampling=BACKGROUND_SAMPLING
            )
    except Exception as e:
        logging.exception("Here is the error")
