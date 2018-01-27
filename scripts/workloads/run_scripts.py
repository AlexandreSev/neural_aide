#!/usr/bin/python
# coding: utf-8

import os

VENV_DIRECTORY = "/home/cedar/asevin/experiment/clean_active_nn/active_nn_venv/bin/activate"
SCRIPTS_DIRECTORY = "/home/cedar/asevin/experiment/clean_active_nn/scripts/"
RESULTS_DIRECOTRY = "/data/asevin/experiment/clean_active_nn/results/"

os.system("source " + VENV_DIRECTORY)

for file_name in os.listdir(SCRIPTS_DIRECTORY):
	if ".py" in file_name:
		name = file_name.split(".")[-2]
		if not os.path.isdir(RESULTS_DIRECOTRY + name):
			os.mkdir(RESULTS_DIRECOTRY + name)
			os.system("screen python " + SCRIPTS_DIRECTORY + file_name)



