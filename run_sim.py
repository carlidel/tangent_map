import argparse
import json
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
                             TrackingConfig, get_config, get_output_config)
from run_functions import (track_coordinates, track_rem, track_stability,
                           track_tangent_map, track_tangent_map_raw)

# create parser
parser = argparse.ArgumentParser(description='Run the simulation')
parser.add_argument('--config', type=str, help='configuration file')

# parse arguments
args = parser.parse_args()

coords, henon, tracking, long_tracking, path, basename = get_config(args.config)

o_tangent_stuff, o_tangent_raw, o_stability, o_coordinates, o_rem = get_output_config(path, basename)

track_tangent_map(coords, henon, tracking, o_tangent_stuff)

track_tangent_map_raw(coords, henon, tracking, o_tangent_raw)

track_stability(coords, henon, long_tracking, o_stability)

track_coordinates(coords, henon, tracking, o_coordinates)

track_rem(coords, henon, tracking, o_rem)
