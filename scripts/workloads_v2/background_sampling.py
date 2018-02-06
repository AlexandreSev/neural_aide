#!/usr/bin/python
# coding: utf-8

from neural_aide import main
import logging
import traceback
import os
from os.path import join as pjoin
from configuration_cedar import configuration

for BACKGROUND_SAMPLING in ["uncertain", "random"]:

    save_dir = pjoin(configuration["RESSOURCES_FOLDER"],
                     "results",
                     "uncertain_sampling_" + BACKGROUND_SAMPLING
               )

    if (not os.path.isdir(save_dir)):
        os.mkdir(save_dir)
    else:
        continue

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
                background_sampling=BACKGROUND_SAMPLING,
                np_seed=i,
                tf_seed=10+i,
                saving_dir=save_dir,
                )
        except Exception as e:
            logging.exception("Here is the error")
        