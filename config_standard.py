import copy
import itertools
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class CoordinateConfig:
    coord1: Literal["x", "px", "y", "py"]
    coord2: Literal["x", "px", "y", "py"]
    coord1_min: float
    coord1_max: float
    coord2_min: float
    coord2_max: float

    samples_per_side: int = 100
    total_samples: int = field(init=False)
    c1_idx: int = field(init=False)
    c2_idx: int = field(init=False)

    def __post_init__(self):
        self.total_samples = self.samples_per_side ** 2

        if self.coord1 == "x":
            self.c1_idx = 0
        elif self.coord1 == "px":
            self.c1_idx = 1
        elif self.coord1 == "y":
            self.c1_idx = 2
        elif self.coord1 == "py":
            self.c1_idx = 3
        else:
            raise ValueError("Invalid coordinate")

        if self.coord2 == "x":
            self.c2_idx = 0
        elif self.coord2 == "px":
            self.c2_idx = 1
        elif self.coord2 == "y":
            self.c2_idx = 2
        elif self.coord2 == "py":
            self.c2_idx = 3
        else:
            raise ValueError("Invalid coordinate")

    def get_variables(self):
        a = np.linspace(self.coord1_min, self.coord1_max, self.samples_per_side)
        b = np.linspace(self.coord2_min, self.coord2_max, self.samples_per_side)
        aa, bb = np.meshgrid(a, b)
        a = aa.flatten()
        b = bb.flatten()
        z = np.zeros_like(a)
        variables = [z, z, z, z]
        variables[self.c1_idx] = a
        variables[self.c2_idx] = b
        return variables
        

@dataclass
class HenonConfig:
    omega_x: float
    omega_y: float

    epsilon: float = 0.0
    mu: float = 0.0
    modulation_kind: str = "sps"
    omega_0: float = np.nan
    barrier: float = 1e4


@dataclass
class TrackingConfig:
    max_iterations: int
    n_samples: int
    sampling_method: Literal["linear", "log", "all"]
    analysis_type: Literal[
        "coordinates",
        "stability",
        "tangent_map",
        "rem",
        "raw_tangent_map",
        "MEGNO"
    ]

    def __post_init__(self):
        if self.sampling_method == "all":
            self.n_samples = self.max_iterations

    def get_samples(self):
        if self.sampling_method == "linear":
            return np.linspace(1, self.max_iterations, self.n_samples, dtype=int)
        elif self.sampling_method == "log":
            return np.logspace(1, np.log10(self.max_iterations), self.n_samples, dtype=int)
        elif self.sampling_method == "all":
            return np.arange(1, self.max_iterations+1, dtype=int)
        else:
            raise ValueError("Invalid sampling method")


@dataclass
class OutputConfig:
    path: str
    basename: str


def f_to_s_without_dot(f):
    return str(f).replace('.', 'd').replace('[', '').replace(']', '').replace(' ', '').replace(',', '_')


def unpack_scans(config: dict)->list[dict]:
    keys = []
    values = []
    if 'scan' in config:
        for key, val in config['scan'].items():
            keys.append(key)
            values.append(val)
    else:
        config["scan_name"] = "default"
        return [config]

    config_list = []
    for vals in itertools.product(*values):
        scan_name = ''    
        config_list.append(copy.deepcopy(config))
        for k, v in zip(keys, vals):
            # split k into k1 and k2 using as delimiter ':'
            k1, k2 = k.split(':')
            config_list[-1][k1][k2] = v
            scan_name += f'{k2}_{f_to_s_without_dot(v)}_'
        config_list[-1].pop('scan')
        config_list[-1]['scan_name'] = scan_name[:-1]

    return config_list


def get_config(config: dict):
    # pass config['coords'] to CoordinateConfig
    coords = CoordinateConfig(
        coord1=config['coords']['coord1'],
        coord2=config['coords']['coord2'],
        coord1_min=config['coords']['coord1_min'],
        coord1_max=config['coords']['coord1_max'],
        coord2_min=config['coords']['coord2_min'],
        coord2_max=config['coords']['coord2_max'],
        samples_per_side=config['coords']['samples_per_side'])

    # pass config['henon'] to HenonConfig
    henon = HenonConfig(
        omega_x=config['henon']['omega_base'][0],
        omega_y=config['henon']['omega_base'][1],
        epsilon=config['henon']['epsilon'],
        mu=config['henon']['mu'])

    # pass config['tracking'] to TrackingConfig
    tracking = TrackingConfig(
        max_iterations=config['tracking']['max_iterations'],
        n_samples=config['tracking']['n_samples'],
        sampling_method=config['tracking']['sampling_method'],
        analysis_type=config['tracking']['analysis_type'])

    long_tracking = TrackingConfig(
        max_iterations=config['tracking']['max_iterations_long'],
        n_samples=config['tracking']['max_iterations_long'],
        sampling_method="all",
        analysis_type="stability")

    # load basename
    basename = config['output']['basename']
    path = config['output']['path']

    return coords, henon, tracking, long_tracking, path, basename


def get_output_config(path, basename, scan_name):
    o_tangent_stuff = OutputConfig(
        path=path,
        basename=f"{basename}{'_' if basename!='' else ''}{scan_name}_tangent_stuff")

    o_tangent_raw = OutputConfig(
        path=path,
        basename=f"{basename}{'_' if basename!='' else ''}{scan_name}_tangent_raw")

    o_stability = OutputConfig(
        path=path,
        basename=f"{basename}{'_' if basename!='' else ''}{scan_name}_stability")

    o_coordinates = OutputConfig(
        path=path,
        basename=f"{basename}{'_' if basename!='' else ''}{scan_name}_coordinates")

    o_rem = OutputConfig(
        path=path,
        basename=f"{basename}{'_' if basename!='' else ''}{scan_name}_rem")

    return o_tangent_stuff, o_tangent_raw, o_stability, o_coordinates, o_rem
