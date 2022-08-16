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


# @njit
def t_map_data(matrices, lyapunov_error, fli_base_x, fli_base_px, fli_base_y, fli_base_py, gali):
    BASE_X = np.array([1.0, 0.0, 0.0, 0.0])
    BASE_PX = np.array([0.0, 0.0, 1.0, 0.0])
    BASE_Y = np.array([0.0, 1.0, 0.0, 0.0])
    BASE_PY = np.array([0.0, 0.0, 0.0, 1.0])

    for i, m in enumerate(matrices):
        lyapunov_error[i] = np.trace(m @ m.T)

        vec_x = m.T @ BASE_X
        vec_px = m.T @ BASE_PX
        vec_y = m.T @ BASE_Y
        vec_py = m.T @ BASE_PY

        fli_base_x[i] = np.log10(np.linalg.norm(vec_x))
        fli_base_px[i] = np.log10(np.linalg.norm(vec_px))
        fli_base_y[i] = np.log10(np.linalg.norm(vec_y))
        fli_base_py[i] = np.log10(np.linalg.norm(vec_py))

        vec_x /= np.linalg.norm(vec_x)
        vec_px /= np.linalg.norm(vec_px)
        vec_y /= np.linalg.norm(vec_y)
        vec_py /= np.linalg.norm(vec_py)

        gali_matrix = np.array([
            vec_x, vec_px, vec_y, vec_py
        ])
        if np.any(np.isnan(gali_matrix)):
            gali[i] = np.nan
        else:
            _, s, _ = np.linalg.svd(gali_matrix)
            gali[i] = np.prod(s)

    return lyapunov_error, fli_base_x, fli_base_px, fli_base_y, fli_base_py, gali


def track_tangent_map(coord: CoordinateConfig, henon: HenonConfig, tracking: TrackingConfig, output: OutputConfig):
    times = tracking.get_samples()
    x, px, y, py = coord.get_variables()

    particles = hm.particles(x, px, y, py)
    matrices = hm.matrix_4d_vector(coord.total_samples, force_cpu=True)
    engine = hm.henon_tracker(
        tracking.max_iterations+1,
        henon.omega_x,
        henon.omega_y,
        henon.modulation_kind,
        henon.omega_0,
        henon.epsilon
    )

    for i in tqdm(range(1, tracking.max_iterations+1)):
        engine.track(particles, 1, henon.mu, henon.barrier)
        matrices.structured_multiply(engine, particles, henon.mu)

        if i in times:
            print(f"Saving time {i}.")
            m = matrices.get_matrix()
            lyapunov_error = np.empty(coord.total_samples)
            fli_base_x = np.empty(coord.total_samples)
            fli_base_px = np.empty(coord.total_samples)
            fli_base_y = np.empty(coord.total_samples)
            fli_base_py = np.empty(coord.total_samples)
            gali = np.empty(coord.total_samples)

            lyapunov_error, fli_base_x, fli_base_px, fli_base_y, fli_base_py, gali = t_map_data(m, lyapunov_error, fli_base_x, fli_base_px, fli_base_y, fli_base_py, gali)

            with h5py.File(f"{output.path}/{output.basename}.h5", "a") as f:
                f.create_dataset(f"{i}/lyapunov_error", data=lyapunov_error, compression="gzip")
                f.create_dataset(f"{i}/fli_base_x", data=fli_base_x, compression="gzip")
                f.create_dataset(f"{i}/fli_base_px", data=fli_base_px, compression="gzip")
                f.create_dataset(f"{i}/fli_base_y", data=fli_base_y, compression="gzip")
                f.create_dataset(f"{i}/fli_base_py", data=fli_base_py, compression="gzip")
                f.create_dataset(f"{i}/gali", data=gali, compression="gzip")

            print(f"Saved time {i}.")


def track_coordinates(coord: CoordinateConfig, henon: HenonConfig, tracking: TrackingConfig, output: OutputConfig):
    times = tracking.get_samples()
    x, px, y, py = coord.get_variables()

    particles = hm.particles(x, px, y, py)
    engine = hm.henon_tracker(
        tracking.max_iterations+1,
        henon.omega_x,
        henon.omega_y,
        henon.modulation_kind,
        henon.omega_0,
        henon.epsilon
    )

    with h5py.File(f"{output.path}/{output.basename}.h5", "w+") as f:
        f.create_dataset(f"x", compression="gzip", shape=(tracking.max_iterations, coord.total_samples), dtype=np.float64)
        f.create_dataset(f"px", compression="gzip", shape=(tracking.max_iterations, coord.total_samples), dtype=np.float64)
        f.create_dataset(f"y", compression="gzip", shape=(tracking.max_iterations, coord.total_samples), dtype=np.float64)
        f.create_dataset(f"py", compression="gzip", shape=(tracking.max_iterations, coord.total_samples), dtype=np.float64)

    for i in tqdm(range(1, tracking.max_iterations+1)):
        engine.track(particles, 1, henon.mu, henon.barrier)

        if i in times:
            print(f"Saving time {i}.")

            with h5py.File(f"{output.path}/{output.basename}.h5", "a") as f:
                f[f"x"][i-1, :] = particles.get_x()
                f[f"px"][i-1, :] = particles.get_px()
                f[f"y"][i-1, :] = particles.get_y()
                f[f"py"][i-1, :] = particles.get_py()

            print(f"Saved time {i}.")


def track_stability(coord: CoordinateConfig, henon: HenonConfig, tracking: TrackingConfig, output: OutputConfig):
    times = tracking.get_samples()
    x, px, y, py = coord.get_variables()

    particles = hm.particles(x, px, y, py)
    engine = hm.henon_tracker(
        tracking.max_iterations+1,
        henon.omega_x,
        henon.omega_y,
        henon.modulation_kind,
        henon.omega_0,
        henon.epsilon
    )
    
    engine.track(particles, tracking.max_iterations, henon.mu, henon.barrier)

    with h5py.File(f"{output.path}/{output.basename}.h5", "a") as f:
        f.create_dataset(f"stability", data=particles.get_steps(), compression="gzip")

    print(f"Saved time {tracking.max_iterations}.")


def track_rem(coord: CoordinateConfig, henon: HenonConfig, tracking: TrackingConfig, output: OutputConfig):
    times = tracking.get_samples()
    x, px, y, py = coord.get_variables()

    particles = hm.particles(x, px, y, py)
    with h5py.File(f"{output.path}/{output.basename}.h5", "a") as f:
        f.create_dataset(f"{0}/x", data=particles.get_x(), compression="gzip")
        f.create_dataset(f"{0}/px", data=particles.get_px(), compression="gzip")
        f.create_dataset(f"{0}/y", data=particles.get_y(), compression="gzip")
        f.create_dataset(f"{0}/py", data=particles.get_py(), compression="gzip")

    engine = hm.henon_tracker(
        tracking.max_iterations+1,
        henon.omega_x,
        henon.omega_y,
        henon.modulation_kind,
        henon.omega_0,
        henon.epsilon
    )

    for t in tqdm(times):
        particles = hm.particles(x, px, y, py)
        engine.track(particles, t, henon.mu, henon.barrier)
        engine.track(particles, t, henon.mu, henon.barrier, inverse=True)

        with h5py.File(f"{output.path}/{output.basename}.h5", "a") as f:
            f.create_dataset(f"{t}/x", data=particles.get_x(), compression="gzip")
            f.create_dataset(f"{t}/px", data=particles.get_px(), compression="gzip")
            f.create_dataset(f"{t}/y", data=particles.get_y(), compression="gzip")
            f.create_dataset(f"{t}/py", data=particles.get_py(), compression="gzip")


def track_tangent_map_raw(coord: CoordinateConfig, henon: HenonConfig, tracking: TrackingConfig, output: OutputConfig):
    times = tracking.get_samples()
    x, px, y, py = coord.get_variables()

    particles = hm.particles(x, px, y, py)

    engine = hm.henon_tracker(
        tracking.max_iterations+1,
        henon.omega_x,
        henon.omega_y,
        henon.modulation_kind,
        henon.omega_0,
        henon.epsilon
    )

    matrices = engine.get_tangent_matrix(particles, henon.mu, reverse=False)
    matrices_rev = engine.get_tangent_matrix(particles, henon.mu, reverse=True)
    print(f"Saving time {0}.")
    with h5py.File(f"{output.path}/{output.basename}.h5", "a") as f:
        f.create_dataset(f"{0}/tangent", data=matrices, compression="gzip")
        f.create_dataset(f"{0}/tangent_rev", data=matrices_rev, compression="gzip")
    print(f"Saved time {0}.")

    for i in tqdm(range(1, tracking.max_iterations+1)):
        engine.track(particles, 1, henon.mu, henon.barrier)
        matrices = engine.get_tangent_matrix(particles, henon.mu, reverse=False)
        matrices_rev = engine.get_tangent_matrix(particles, henon.mu, reverse=True)

        if i in times:
            print(f"Saving time {i}.")
            with h5py.File(f"{output.path}/{output.basename}.h5", "a") as f:
                f.create_dataset(f"{i}/tangent", data=matrices, compression="gzip")
                f.create_dataset(f"{i}/tangent_rev", data=matrices_rev, compression="gzip")
            print(f"Saved time {i}.")
