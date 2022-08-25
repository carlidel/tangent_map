import argparse
import json
import pickle
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
from postprocessings import frequency_map_analysis

# create parser
parser = argparse.ArgumentParser(description='Run the simulation')
parser.add_argument('--config', type=str, help='configuration file')
parser.add_argument('--n-samples', type=int, help='number of samples', default=1000)

# parse arguments
args = parser.parse_args()

coords, henon, tracking, long_tracking, path, basename = get_config(args.config)

_, _, _, o_coordinates, _ = get_output_config(path, basename)

t_half_list, fma_values = frequency_map_analysis(coords, henon, tracking, o_coordinates, n_samples=args.n_samples)

with open(f"{path}/{basename}{'_' if basename!='' else ''}FMA.pkl", "wb") as f:
    pickle.dump((t_half_list, fma_values), f)
