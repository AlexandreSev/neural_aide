#!/usr/bin/python
# coding: utf-8

from neural_aide import main
import logging

# For a single query run:
RESSOURCES_FOLDER = "/data/asevin/experiment/clean_nn_active/"

try:
    # Random QDB without using main weights
    main.run_experiment_with_sdss(RESSOURCES_FOLDER, qdb=True, random=True,
                        shapes=[32, 32, 1], include_background=True,
                        evolutive_small=True, nb_biased_epoch=2000,
                        use_main_weights=False, display=False, save_plot=False,
                        query=2)
except:
    logging.exception("Here is the error of query 2, qdb main !")

try:
    # Random QDB using main weights
    main.run_experiment_with_sdss(RESSOURCES_FOLDER, qdb=True, random=True,
                        shapes=[32, 32, 1], include_background=True,
                        evolutive_small=True, nb_biased_epoch=2000,
                        use_main_weights=True, display=False, save_plot=False,
                        query=2)
except:
    logging.exception("Here is the error of query 2, qdb not main !")

try:
    # Uncertainty sampling
    main.run_experiment_with_sdss(RESSOURCES_FOLDER, qdb=False, random=True,
                        shapes=[32, 32, 1], include_background=True,
                        evolutive_small=True, nb_biased_epoch=2000,
                        use_main_weights=True, display=False, save_plot=False,
                        query=2)
except:
    logging.exception("Here is the error of query 2, uncertainty !")

try:
    # Random QDB without using main weights
    main.run_experiment_with_sdss(RESSOURCES_FOLDER, qdb=True, random=True,
                        shapes=[32, 32, 1], include_background=True,
                        evolutive_small=True, nb_biased_epoch=2000,
                        use_main_weights=False, display=False, save_plot=False,
                        query=3)
except:
    logging.exception("Here is the error of query 3, qdb using main!")

try:
    # Random QDB using main weights
    main.run_experiment_with_sdss(RESSOURCES_FOLDER, qdb=True, random=True,
                        shapes=[32, 32, 1], include_background=True,
                        evolutive_small=True, nb_biased_epoch=2000,
                        use_main_weights=True, display=False, save_plot=False,
                        query=3)
except:
    logging.exception("Here is the error of query 3, qdb not main!")

try:
    # Uncertainty sampling
    main.run_experiment_with_sdss(RESSOURCES_FOLDER, qdb=False, random=True,
                        shapes=[32, 32, 1], include_background=True,
                        evolutive_small=True, nb_biased_epoch=2000,
                        use_main_weights=True, display=False, save_plot=False,
                        query=3)
except:
    logging.exception("Here is the error of query 3, uncertainty!")
