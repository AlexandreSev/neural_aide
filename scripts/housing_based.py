#!/usr/bin/python
# coding: utf-8

from neural_aide import main
import logging
import traceback
import sys

RESSOURCES_FOLDER = "/Users/alex/Documents/LIX-PHD/experiments/clean_active_nn"
QDB = True
RANDOM = True
SHAPES = [128, 1]
INCLUDE_BACKGROUND = True
EVOLUTIVE_SMALL = True
NB_BIASED_EPOCH = 2000
USE_MAIN_WEIGHTS = False
BIASED_LR = 0.001
TSM = False
TSM_LIM = None
REDUCE_FACTOR = None

try:
    main.run_experiment_with_housing(RESSOURCES_FOLDER, qdb=QDB, random=RANDOM,
                        shapes=SHAPES, include_background=INCLUDE_BACKGROUND,
                        evolutive_small=EVOLUTIVE_SMALL,
                        nb_biased_epoch=NB_BIASED_EPOCH,
                        use_main_weights=USE_MAIN_WEIGHTS,
                        biased_lr=BIASED_LR, tsm=TSM,
                        tsm_lim=TSM_LIM, reduce_factor=REDUCE_FACTOR)
except Exception as e:
    logging.exception("Here is the error")