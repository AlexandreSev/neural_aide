#!/usr/bin/python
# coding: utf-8

from neural_aide import main
import logging

# For a single query run:
RESSOURCES_FOLDER = "/data/asevin/experiment/clean_active_nn/"
query = 1

for qdb in [True, False]:
    for shapes in [[4, 1], [8, 1], [16, 1], [32, 1], [8, 4, 1], [16, 8, 1], [32, 16, 1]]:
        for evolutive_small in [True, False]:
            for use_main_weights in [True, False]:
                for biased_lr in [0.0001, 0.001, 0.01]:
                    for reduce_factor in [None, "evolutive"]:

                        try:
                            # Random QDB without using main weights
                            main.run_experiment_with_sdss(
                                RESSOURCES_FOLDER,
                                qdb=qdb,
                                random=True,
                                shapes=shapes,
                                include_background=True,
                                evolutive_small=evolutive_small,
                                nb_biased_epoch=2000,
                                use_main_weights=use_main_weights,
                                display=False,
                                save_plot=False,
                                query=query,
                                biased_lr=biased_lr,
                                tsm=False,
                                reduce_factor=reduce_factor)
                        except:
                            logging.exception("Here is the error, qdb main !")
