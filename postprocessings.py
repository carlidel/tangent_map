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


@njit
def birkhoff_weights(n):
    weights = np.arange(n, dtype=np.float64)
    weights /= n
    weights = np.exp(-1/(weights * (1 - weights)))
    return weights / np.sum(weights)


@njit
def birkhoff_tune(x, px):
    assert(x.size == px.size)
    n = x.size

    z = x + 1j * px
    theta = np.angle(z)
    d_theta = np.diff(theta)
    d_theta[d_theta < 0] += 2 * np.pi

    weights = birkhoff_weights(n - 1)
    final_sum = np.sum(weights * d_theta)
    return 1 - final_sum / (2 * np.pi)


@njit
def broadcast_matmul(mat, vec):
    result = np.empty_like(vec)
    for i in range(mat.shape[0]):
        result[i] = mat[i] @ vec[i]
    return result


def compute_RE(config, coord: CoordinateConfig, henon: HenonConfig, tracking: TrackingConfig, output: OutputConfig):
    max_time = tracking.max_iterations

    f = h5py.File(f"{output.path}/{output.basename}.h5", 'r')
    
    fm = []
    fm_r = []
    for i in tqdm(range(max_time+1)):
        fm.append(f[f"{i}/tangent"][:])
        fm_r.append(f[f"{i}/tangent_rev"][:])

    random = np.random.normal(size=(max_time * 2 * 4)).reshape(max_time * 2, 4)
    for i in range(random.shape[0]):
        random[i] /= np.linalg.norm(random[i])

    sampling_times = config["postprocessing"]["full_samples"]

    re_values = np.zeros((len(sampling_times), coord.total_samples)) * np.nan
    
    for idx, t in tqdm(enumerate(sampling_times), total=len(sampling_times)):
        bf = np.array([random[-1] for i in range(coord.total_samples)])

        for i in range(t):
            bf = broadcast_matmul(fm[t][:], bf) + random[None, t]
        bf_rev = bf.copy()
        for i in (range(t)):
            bf_rev = broadcast_matmul(fm_r[t-i-1][:], bf_rev) + random[None, t+i]

        for i, r in enumerate(bf_rev):
            re_values[idx, i] = np.linalg.norm(r)

    f.close()
    return re_values


def compute_RE_full(config, coord: CoordinateConfig, henon: HenonConfig, tracking: TrackingConfig, output: OutputConfig):
    max_time = tracking.max_iterations

    f = h5py.File(f"{output.path}/{output.basename}.h5", 'r')
    
    fm = []
    fm_r = []
    for i in tqdm(range(max_time+1)):
        fm.append(f[f"{i}/tangent"][:])
        fm_r.append(f[f"{i}/tangent_rev"][:])

    random = np.random.normal(size=(max_time * 2 * 4)).reshape(max_time * 2, 4)
    for i in range(random.shape[0]):
        random[i] /= np.linalg.norm(random[i])

    sampling_points = config["postprocessing"]["precise_initial_conditions"]

    re_values = np.zeros((max_time, len(sampling_points))) * np.nan
    
    for t in tqdm(range(max_time)):
        bf = np.array([random[-1] for i in range(len(sampling_points))])

        for i in range(t):
            bf = broadcast_matmul(fm[t][sampling_points], bf) + random[None, t]
        bf_rev = bf.copy()
        for i in (range(t)):
            bf_rev = broadcast_matmul(fm_r[t-i-1][sampling_points], bf_rev) + random[None, t+i]

        for i, r in enumerate(bf_rev):
            re_values[t, i] = np.linalg.norm(r)

    f.close()
    return re_values


def compute_MEGNO(coord: CoordinateConfig, henon: HenonConfig, tracking: TrackingConfig, output: OutputConfig):
    f = h5py.File(f"{output.path}/{output.basename}.h5", 'r')
    
    max_time = tracking.max_iterations

    le_value_list = []
    for i in tqdm(range(max_time)):
        if i == 0:
            matrices = f[f"{i}/tangent"][:]
        else:
            matrices = f[f"{i}/tangent"][:]@matrices
        le_value = (np.sqrt(np.trace(matrices @ np.transpose(matrices, axes=[0,2,1]), axis1=1, axis2=2)))
        le_value_list.append(le_value)
    le_value_list = np.array(le_value_list)

    f.close()

    y_middle = []
    for i in tqdm(range(1, max_time)):
        val = np.log10(le_value_list[i]/le_value_list[i-1]) * i
        if i == 1:
            y_middle.append(val)
        else:
            y_middle.append(y_middle[-1] + val)
    
    y_vals = []
    for i in range(1, max_time):
        y_vals.append(np.mean(le_value_list[:i], axis=0))
    y_vals = np.array(y_vals)

    return y_vals


def frequency_map_analysis(coord: CoordinateConfig, henon: HenonConfig, tracking: TrackingConfig, output: OutputConfig, n_samples=1000):
    f = h5py.File(f"{output.path}/{output.basename}.h5", 'r')
    max_time = tracking.max_iterations
    times = np.linspace(2, max_time, n_samples, dtype=int)
    t_half_list = times // 2

    fm = {
        "x" : f["x"][:],
        "y" : f["y"][:],
        "px" : f["px"][:],
        "py" : f["py"][:],
    }
    f.close()

    fma_data = np.zeros((len(times), coord.total_samples)) * np.nan
    for i in tqdm(range(coord.total_samples)):
        x_data = fm["x"][:, i]
        px_data = fm["px"][:, i]
        y_data = fm["y"][:, i]
        py_data = fm["py"][:, i]
        for j, t_half in enumerate(t_half_list):
            tune_1_x = birkhoff_tune(x_data[:t_half], px_data[:t_half])
            tune_1_y = birkhoff_tune(y_data[:t_half], py_data[:t_half])
            tune_2_x = birkhoff_tune(x_data[t_half:t_half*2], px_data[t_half:t_half*2])
            tune_2_y = birkhoff_tune(y_data[t_half:t_half*2], py_data[t_half:t_half*2])
            fma_data[j, i] = np.sqrt((tune_1_x - tune_2_x)**2 + (tune_1_y - tune_2_y)**2)

    return t_half_list, fma_data
