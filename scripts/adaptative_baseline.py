#!/usr/bin/python
# coding: utf-8

from neural_aide.adaptive import adaptive_main as main
import logging
import traceback
from os.path import join as pjoin
from configuration_local import configuration

try:
    main.run_experiment_with_sdss(
        configuration["RESSOURCES_FOLDER"],
        qdb=configuration["QDB"],
        random=configuration["RANDOM"],
        shapes=[16, 1],
        include_background=configuration["INCLUDE_BACKGROUND"],
        evolutive_small=configuration["EVOLUTIVE_SMALL"],
        nb_biased_epoch=configuration["NB_BIASED_EPOCH"],
        use_main_weights=True,
        display=configuration["DISPLAY"],
        save_plot=True,
        query=1,
        biased_lr=configuration["BIASED_LR"],
        tsm=configuration["TSM"],
        pltlim=configuration["PLTLIM"],
        tsm_lim=configuration["TSM_LIM"],
        reduce_factor=configuration["REDUCE_FACTOR"],
        pool_size=None,
        main_lr=configuration["MAIN_LR"],
        nn_activation=configuration["ACTIVATION"],
        nn_loss=configuration["LOSS"],
        background_sampling=configuration["BACKGROUND_SAMPLING"],
        np_seed=42,
        tf_seed=7,
        saving_dir=pjoin(configuration["RESSOURCES_FOLDER"],
                         "results", "baseline"),
        )
except Exception as e:
    logging.exception("Here is the error")



