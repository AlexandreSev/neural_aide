#!/usr/bin/python
# coding: utf-8

from neural_aide import main
import logging
import traceback
import os
from os.path import join as pjoin
from configuration_local import configuration

translation_table = {
    "evolutive": "evolutive",
    "evolutive4": "evolutive4",
    "evolutive8": "evolutive8",
    "evolutive16": "evolutive16",
    "none": None,
    "noreduction": 1,
}

for REDUCE_FACTOR in ["evolutive", "none", "noreduction", "evolutive4",
                      "evolutive8", "evolutive16"]:

    for NB_BACKGROUND_POINTS in [0.1, 0.2, 0.5, 1, 2, 5, 10]:

        save_dir = pjoin(configuration["RESSOURCES_FOLDER"],
                         "results",
                         ("reduce_factor_" + REDUCE_FACTOR +
                         "_evolutive_small_True_nb_background_points_" +
                         str(NB_BACKGROUND_POINTS)),
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
                    evolutive_small=True,
                    nb_background_points=NB_BACKGROUND_POINTS,
                    nb_biased_epoch=configuration["NB_BIASED_EPOCH"],
                    use_main_weights=configuration["USE_MAIN_WEIGHTS"],
                    display=configuration["DISPLAY"],
                    save_plot=configuration["SAVE_PLOT"],
                    query=configuration["QUERY"],
                    biased_lr=configuration["BIASED_LR"],
                    tsm=configuration["TSM"],
                    pltlim=configuration["PLTLIM"],
                    tsm_lim=configuration["TSM_LIM"],
                    reduce_factor=translation_table[REDUCE_FACTOR],
                    pool_size=configuration["POOL_SIZE"],
                    main_lr=configuration["MAIN_LR"],
                    nn_activation=configuration["ACTIVATION"],
                    nn_loss=configuration["LOSS"],
                    background_sampling=configuration["BACKGROUND_SAMPLING"],
                    np_seed=i,
                    tf_seed=10+i,
                    saving_dir=save_dir,
                    )
            except Exception as e:
                print(e)
                logging.exception("Here is the error")

    for NB_BACKGROUND_POINTS in [10, 50, 100, 200, 500]:

        save_dir = pjoin(configuration["RESSOURCES_FOLDER"],
                         "results",
                         ("reduce_factor_" + REDUCE_FACTOR +
                         "_evolutive_small_False_nb_background_points_" +
                         str(NB_BACKGROUND_POINTS)),
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
                    evolutive_small=False,
                    nb_background_points=NB_BACKGROUND_POINTS,
                    nb_biased_epoch=configuration["NB_BIASED_EPOCH"],
                    use_main_weights=configuration["USE_MAIN_WEIGHTS"],
                    display=configuration["DISPLAY"],
                    save_plot=configuration["SAVE_PLOT"],
                    query=configuration["QUERY"],
                    biased_lr=configuration["BIASED_LR"],
                    tsm=configuration["TSM"],
                    pltlim=configuration["PLTLIM"],
                    tsm_lim=configuration["TSM_LIM"],
                    reduce_factor=translation_table[REDUCE_FACTOR],
                    pool_size=configuration["POOL_SIZE"],
                    main_lr=configuration["MAIN_LR"],
                    nn_activation=configuration["ACTIVATION"],
                    nn_loss=configuration["LOSS"],
                    background_sampling=configuration["BACKGROUND_SAMPLING"],
                    np_seed=i,
                    tf_seed=10+i,
                    saving_dir=save_dir,
                    )
            except Exception as e:
                print(e)
                logging.exception("Here is the error")
            