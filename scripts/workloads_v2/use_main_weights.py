#!/usr/bin/python
# coding: utf-8

from neural_aide import main
import logging
import traceback
import os
from os.path import join as pjoin
from configuration_cedar import configuration

for USE_MAIN_WEIGHTS in [True, False]:

    save_dir = pjoin(configuration["RESSOURCES_FOLDER"],
                     "results",
                     "use_main_weights_" + USE_MAIN_WEIGHTS
               )

    if (not os.path.isdir(save_dir)):
        os.mkdir(save_dir)
    else:
        pass

    for i in range(10):
        try:
            main.run_experiment_with_sdss(
            	configuration["RESSOURCES_FOLDER"],
                qdb=configuration["QDB"],
                random=configuration["RANDOM"],
                shapes=configuration["SHAPES"],
                include_background=configuration["INCLUDE_BACKGROUND"],
                evolutive_small=configuration["EVOLUTIVE_SMALL"],
                nb_biased_epoch=configuration["NB_BIASED_EPOCH"],
                use_main_weights=USE_MAIN_WEIGHTS,
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
                np_seed=42,
                tf_seed=7,
                saving_dir=save_dir,
                )
        except Exception as e:
            logging.exception("Here is the error")
        