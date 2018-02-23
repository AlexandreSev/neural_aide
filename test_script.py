#!/usr/bin/python
# coding: utf-8

from neural_aide import main
import logging
import traceback
from os.path import join as pjoin
from configuration_test import configuration


main.run_experiment_with_sdss(
    configuration["RESSOURCES_FOLDER"],
    qdb=configuration["QDB"],
    random=configuration["RANDOM"],
    shapes=configuration["SHAPES"],
    include_background=configuration["INCLUDE_BACKGROUND"],
    evolutive_small=configuration["EVOLUTIVE_SMALL"],
    nb_biased_epoch=configuration["NB_BIASED_EPOCH"],
    nb_background_points=configuration["NB_BACKGROUND_POINTS"],
    use_main_weights=configuration["USE_MAIN_WEIGHTS"],
    display=configuration["DISPLAY"],
    save_plot=configuration["SAVE_PLOT"],
    query=configuration["QUERY"],
    biased_lr=configuration["BIASED_LR"],
    tsm=configuration["TSM"],
    pltlim=configuration["PLTLIM"],
    tsm_lim=configuration["TSM_LIM"],
    reduce_factor=configuration["REDUCE_FACTOR"],
    pool_size=configuration["POOL_SIZE"],
    main_lr=configuration["MAIN_LR"],
    nn_activation=configuration["ACTIVATION"],
    nn_loss=configuration["LOSS"],
    background_sampling=configuration["BACKGROUND_SAMPLING"],
    np_seed=configuration["NP_SEED"],
    tf_seed=configuration["TF_SEED"],
    saving_dir=".",
    max_iter=configuration["MAX_ITER"],
    log_into_file=configuration["LOG_INTO_FILE"]
    )


