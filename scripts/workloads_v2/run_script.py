#!/usr/bin/python
# coding: utf-8

import os

for file_name in os.listdir("/home/cedar/asevin/experiment/clean_active_nn/scripts/workloads_v2"):
    if ".py" in file_name:
        print(file_name)
        os.system("python " + os.path.join(
            "/home/cedar/asevin/experiment/clean_active_nn/scripts/workloads_v2",
            file_name))
