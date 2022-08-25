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
from postprocessings import compute_RE

# create parser
parser = argparse.ArgumentParser(description='Run the simulation')
parser.add_argument('--config', type=str, help='configuration file')

# parse arguments
args = parser.parse_args()

coords, henon, tracking, long_tracking, path, basename = get_config(args.config)

_, o_tangent_raw, _, _, _ = get_output_config(path, basename)

re_values = compute_RE(coords, henon, tracking, o_tangent_raw)

with open(f"{path}/{basename}{'_' if basename!='' else ''}RE.pkl", "wb") as f:
    pickle.dump(re_values, f)
