#!/usr/bin/python
# coding: utf-8

import os

for file_name in os.listdir("/home/cedar/asevin/experiment/clean_active_nn/scripts/"):
	if ".py" in file_name:
		os.system("python " + os.path.join(
			"/home/cedar/asevin/experiment/clean_active_nn/scripts/",
			file_name))
		