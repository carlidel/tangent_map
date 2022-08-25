from dataclasses import dataclass, field
from random import shuffle
from typing import Literal

import h5py
import henon_map_cpp as hm
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from numba import njit
from tqdm import tqdm

from config_standard import (CoordinateConfig, HenonConfig, OutputConfig,
                             TrackingConfig)
from run_functions import (track_coordinates, track_rem, track_stability,
                           track_tangent_map, track_tangent_map_raw)

coords = CoordinateConfig(
    coord1="x",
    coord2="y",
    coord1_min=0.0,
    coord1_max=0.25,
    coord2_min=0.0,
    coord2_max=0.25,
    samples_per_side=100)

henon = HenonConfig(
    omega_x=0.31,
    omega_y=0.32,
    epsilon=64.0,
    mu=0.1)

tracking = TrackingConfig(
    max_iterations=10000,
    n_samples=10000,
    sampling_method="all",
    analysis_type="tangent_map")

# output = OutputConfig(
#     path="/home/HPC/camontan/turchetti_paper/output",
#     basename="ter_tangent_stuff")

# track_tangent_map(coords, henon, tracking, output)

# output = OutputConfig(
#     path="/home/HPC/camontan/turchetti_paper/output",
#     basename="ter_tangent_raw")

# track_tangent_map_raw(coords, henon, tracking, output)

# output = OutputConfig(
#     path="/home/HPC/camontan/turchetti_paper/output",
#     basename="ter_stability")

# track_stability(coords, henon, tracking, output)

output = OutputConfig(
    path="/home/HPC/camontan/turchetti_paper/output",
    basename="ter_coordinates")

track_coordinates(coords, henon, tracking, output)

output = OutputConfig(
    path="/home/HPC/camontan/turchetti_paper/output",
    basename="ter_rem")

track_rem(coords, henon, tracking, output)
