#!/usr/bin/python
# coding: utf-8

from neural_aide import main
import logging
import traceback
import sys

RESSOURCES_FOLDER = "/Users/alex/Documents/LIX-PHD/experiments/clean_active_nn"
QDB = True
RANDOM = True
SHAPES = [32, 1]
INCLUDE_BACKGROUND = True
EVOLUTIVE_SMALL = True
NB_BIASED_EPOCH = 2000
USE_MAIN_WEIGHTS = False
DISPLAY = False
SAVE_PLOT = True
QUERY = 1
BIASED_LR = 0.001
TSM = False
PLTLIM = None
TSM_LIM = None
REDUCE_FACTOR = None
POOL_SIZE = None
NP_SEED = 42
TF_SEED = 42


try:
    for i in range(50):
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
            np_seed=NP_SEED,
            tf_seed=TF_SEED,
            )
except Exception as e:
    logging.exception("Here is the error")