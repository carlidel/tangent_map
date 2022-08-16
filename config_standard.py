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
