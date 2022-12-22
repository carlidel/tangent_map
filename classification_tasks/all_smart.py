import argparse
import itertools
import os
import pickle
import sys

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

sys.path.insert(0, "/home/HPC/camontan/turchetti_paper/")

from clustering_scripts import *

## Create parser
parser = argparse.ArgumentParser(
    description="Plot the results of the classification task."
)
parser.add_argument("--idx", required=True, type=int, help="Index of the task.")

args = parser.parse_args()
idx = args.idx


OUTDIR = "/home/HPC/camontan/turchetti_paper/output/focused_scan/"
OUTDIR2 = "/home/HPC/camontan/turchetti_paper/output/focused_scan_2/"
OUTDIR3 = "/home/HPC/camontan/turchetti_paper/output/"

TUNES = ["scan_28_31_", "scan_168_201_"]
TUNES_NAMES = ["(0.28, 0.31)", "(0.168, 0.201)"]
EXTENTS = [(0, 0.45, 0, 0.45), (0, 0.6, 0, 0.6)]
EPSILONS = ["epsilon_0d0_", "epsilon_32d0_"]
EPSILONS_NAMES = ["0.0", "32.0"]
MUS = ["mu_0d0_", "mu_0d5_"]
MUS_NAMES = ["0.0", "0.5"]

SAMPLES_PER_SIDE = [300]

# create a list of all the products of the above lists
# this will be used to loop over all the different files
# and load the data
all_combinations = list(
    itertools.product(((t, s) for t, s in zip(TUNES, EXTENTS)), EPSILONS, MUS)
)
all_combinations = list(itertools.product(TUNES, EPSILONS, MUS))
name_combinations = list(itertools.product(TUNES_NAMES, EPSILONS_NAMES, MUS_NAMES))
samples = SAMPLES_PER_SIDE[0]


def get_hdf5_files(tune, epsilon, mu, samples):
    f_stab = h5py.File(
        os.path.join(
            OUTDIR, tune + epsilon + mu + "analysis_type_stability_stability.h5"
        ),
        "r",
    )
    f_lyap = h5py.File(
        os.path.join(
            OUTDIR, tune + epsilon + mu + "analysis_type_tangent_map_tangent_stuff.h5"
        ),
        "r",
    )
    f_rem = h5py.File(
        os.path.join(OUTDIR, tune + epsilon + mu + "analysis_type_rem_rem.h5"), "r"
    )
    f_tune = h5py.File(
        os.path.join(OUTDIR2, tune + epsilon + mu + "analysis_type_tune_tune.h5"), "r"
    )
    f_birkhoff = h5py.File(
        os.path.join(
            OUTDIR2,
            tune
            + epsilon
            + mu
            + "analysis_type_lyapunov_birkhoff_lyapunov_birkhoff.h5",
        ),
        "r",
    )
    f_megno = h5py.File(
        os.path.join(OUTDIR2, tune + epsilon + mu + "analysis_type_megno_megno.h5"), "r"
    )

    return f_stab, f_lyap, f_rem, f_tune, f_birkhoff, f_megno


f_stab_list = []
f_lyap_list = []
f_rem_list = []
f_tune_list = []
f_birkhoff_list = []
f_megno_list = []

for combo in tqdm(all_combinations):
    tune, epsilon, mu = combo[0], combo[1], combo[2]
    f_stab, f_lyap, f_rem, f_tune, f_birkhoff, f_megno = get_hdf5_files(
        tune, epsilon, mu, samples
    )
    f_stab_list.append(f_stab)
    f_lyap_list.append(f_lyap)
    f_rem_list.append(f_rem)
    f_tune_list.append(f_tune)
    f_birkhoff_list.append(f_birkhoff)
    f_megno_list.append(f_megno)

f_stab, f_lyap, f_rem, f_tune, f_birkhoff, f_megno = list(
    zip(
        f_stab_list, f_lyap_list, f_rem_list, f_tune_list, f_birkhoff_list, f_megno_list
    )
)[idx]

############ GROUND TRUTH ######################################################
print("Loading ground truth...")

data = f_stab["stability"][:]

initial_x = f_stab["initial/x"][:]
initial_y = f_stab["initial/y"][:]
initial_px = f_stab["initial/px"][:]
initial_py = f_stab["initial/py"][:]

mask = np.log10(data) == 8
print("mask_shape", mask.shape)

ground_truth_data = f_lyap["100000000/lyapunov_error"][:]
ground_truth_data[np.isinf(ground_truth_data)] = np.nanmax(
    ground_truth_data[~np.isinf(ground_truth_data)]
)
ground_truth_data[mask & (np.isnan(ground_truth_data))] = np.nanmax(
    ground_truth_data[~np.isinf(ground_truth_data)]
)
ground_truth_data[~mask] = np.nan
ground_truth_data = np.log10(ground_truth_data)

gt_thesh = find_threshold_smart(np.log10(ground_truth_data[mask]), where_chaos="higher")
ground_truth = np.log10(ground_truth_data) > gt_thesh

############ LYAPUNOV ##########################################################
print("Loading lyapunov...")

times = []
lyapunov_data = []

for key in f_lyap.keys():
    if not key.isdigit():
        continue
    times.append(int(key))
    lyapunov_data.append(f_lyap[key]["lyapunov_error"][:])

# sort times and lyapunov data
times, lyapunov_data = zip(*sorted(zip(times, lyapunov_data)))

# convert to numpy arrays
times = np.array(times)
lyapunov_data = np.array(lyapunov_data)

lyapunov_thresholds = []
lyapunov_post_data = []
lyapunov_guesses = []
lyapunov_scores = []
for t, data in zip(times, lyapunov_data):
    data[mask & (np.isnan(data))] = np.nanmax(data[mask])
    data = np.log10(data)
    data = np.log10(data)
    data[mask & (np.isnan(data))] = np.nanmax(data[mask])
    data[mask & (np.isinf(data))] = np.nanmax(data[mask & (~np.isinf(data))])
    data[~mask] = np.nan
    lyapunov_post_data.append(data)
    lyapunov_thresholds.append(find_threshold_smart(data[mask], where_chaos="higher"))
    guess = data > lyapunov_thresholds[-1]
    lyapunov_guesses.append(guess)
    lyapunov_scores.append(classify_data(ground_truth[mask], guess[mask]))

############ MEGNO #############################################################
print("Loading megno...")

times = []
megno_data = []

for key in f_megno.keys():
    if not key.isdigit():
        continue
    times.append(int(key))
    megno_data.append(f_megno[key]["megno"][:])

# sort times and megno data
times, megno_data = zip(*sorted(zip(times, megno_data)))

# convert to numpy arrays
times = np.array(times)
megno_data = np.array(megno_data)

megno_thresholds = []
megno_post_data = []
megno_guesses = []
megno_scores = []
for t, data in zip(times, megno_data):
    data[mask & (np.isnan(data))] = np.nanmax(data[mask])
    data = np.log10(data)
    data[mask & (np.isnan(data))] = np.nanmax(data[mask])
    data[mask & (np.isinf(data))] = np.nanmax(data[mask & (~np.isinf(data))])
    data[~mask] = np.nan
    megno_post_data.append(data)
    megno_thresholds.append(find_threshold_smart(data[mask], where_chaos="higher"))
    guess = data > megno_thresholds[-1]
    megno_guesses.append(guess)
    megno_scores.append(classify_data(ground_truth[mask], guess[mask]))

############ FLI ###############################################################
print("Loading fli...")

times = []
fli_x_data = []

for key in f_lyap.keys():
    if not key.isdigit():
        continue
    times.append(int(key))
    fli_x_data.append(f_lyap[key]["fli_base_x"][:])

# sort times and lyapunov data
times, fli_x_data = zip(*sorted(zip(times, fli_x_data)))

# convert to numpy arrays
times = np.array(times)
fli_x_data = np.array(fli_x_data)

fli_x_thresholds = []
fli_x_post_data = []
fli_x_guesses = []
fli_x_scores = []
for t, data in zip(times, fli_x_data):
    data[mask & (np.isnan(data))] = np.nanmax(data[mask])
    # data = np.log10(data)
    data[mask & np.isinf(data)] = np.nanmax(data[~np.isinf(data)])
    fli_x_post_data.append(data)

    fli_x_thresholds.append(find_threshold_smart(data[mask], where_chaos="higher"))
    guess = data > fli_x_thresholds[-1]
    fli_x_guesses.append(guess)
    fli_x_scores.append(classify_data(ground_truth[mask], guess[mask]))

############ GALI ##############################################################
print("Loading gali...")

times = []
gali_data = []

for key in f_lyap.keys():
    if not key.isdigit():
        continue
    times.append(int(key))
    gali_data.append(f_lyap[key]["gali"][:])

# sort times and lyapunov data
times, gali_data = zip(*sorted(zip(times, gali_data)))

# convert to numpy arrays
times = np.array(times)
gali_data = np.array(gali_data)

gali_thresholds = []
gali_post_data = []
gali_guesses = []
gali_scores = []
for t, data in zip(times, gali_data):
    data[mask & (np.isnan(data))] = np.nanmin(data[mask])
    data = np.log10(data)
    data_to_save = data.copy()
    data[mask & np.isnan(data)] = np.nanmin(data[~np.isinf(data)])
    data[mask & np.isinf(data)] = np.nanmin(data[~np.isinf(data)])
    data_to_save[np.isnan(data_to_save)] = -64
    data_to_save[np.isinf(data_to_save)] = -64
    data[~mask] = np.nan
    data_to_save[~mask] = np.nan
    gali_post_data.append(data_to_save)

    data_clone = data[mask].copy()
    data_min = np.nanmin(data_clone)
    if idx != 0 and idx != 1 and idx != 4 and idx != 5:
        data_clone[data_clone <= data_min] = np.nan

    gali_thresholds.append(find_threshold_smart(data_clone, where_chaos="gali"))
    guess = data < gali_thresholds[-1]
    gali_guesses.append(guess)
    gali_scores.append(classify_data(ground_truth[mask], guess[mask]))

############# FMA ##############################################################
print("Loading fma...")

times_fma = []
fma_fft_data = []
fma_birkhoff_data = []

for key in f_tune["0"].keys():
    if not key.isdigit():
        continue
    if not (key in f_tune):
        continue
    times_fma.append(int(key) * 2)

    fma_fft_data.append(
        np.sqrt(
            (
                f_tune["0"][key]["tune_x_fft"][:]
                - f_tune[key][str(int(key) * 2)]["tune_x_fft"][:]
            )
            ** 2
            + (
                f_tune["0"][key]["tune_y_fft"][:]
                - f_tune[key][str(int(key) * 2)]["tune_y_fft"][:]
            )
            ** 2
        )
    )
    fma_birkhoff_data.append(
        np.sqrt(
            (
                f_tune["0"][key]["tune_x_birkhoff"][:]
                - f_tune[key][str(int(key) * 2)]["tune_x_birkhoff"][:]
            )
            ** 2
            + (
                f_tune["0"][key]["tune_y_birkhoff"][:]
                - f_tune[key][str(int(key) * 2)]["tune_y_birkhoff"][:]
            )
            ** 2
        )
    )

# sort times_fma and lyapunov data
times_fma, fma_fft_data, fma_birkhoff_data = zip(
    *sorted(zip(times_fma, fma_fft_data, fma_birkhoff_data), key=lambda x: x[0])
)

# convert to numpy arrays
times_fma = np.array(times_fma)
fma_fft_data = np.array(fma_fft_data)
fma_birkhoff_data = np.array(fma_birkhoff_data)

fma_fft_thresholds = []
fma_fft_post_data = []
fma_fft_guesses = []
fma_fft_scores = []
for t, data in zip(times_fma, fma_fft_data):
    data[mask & (np.isnan(data))] = np.nanmin(data[mask])
    data = np.log10(data)
    data[mask & np.isnan(data)] = np.nanmin(data[~np.isinf(data)])
    data[mask & np.isinf(data)] = np.nanmin(data[~np.isinf(data)])
    data[~mask] = np.nan

    fma_fft_post_data.append(data)

    fma_fft_thresholds.append(find_threshold_smart_v2(data[mask], where_chaos="higher"))
    guess = data > fma_fft_thresholds[-1]
    fma_fft_guesses.append(guess)
    fma_fft_scores.append(classify_data(ground_truth[mask], guess[mask]))

fma_birkhoff_thresholds = []
fma_birkhoff_post_data = []
fma_birkhoff_guesses = []
fma_birkhoff_scores = []
for t, data in zip(times_fma, fma_birkhoff_data):
    data[mask & (np.isnan(data))] = np.nanmin(data[mask])
    data = np.log10(data)
    data[mask & np.isnan(data)] = np.nanmin(data[~np.isinf(data)])
    data[mask & np.isinf(data)] = np.nanmin(data[~np.isinf(data)])
    data[~mask] = np.nan

    fma_birkhoff_post_data.append(data)

    fma_birkhoff_thresholds.append(
        find_threshold_smart_v2(data[mask], where_chaos="higher")
    )
    guess = data > fma_birkhoff_thresholds[-1]
    fma_birkhoff_guesses.append(guess)
    fma_birkhoff_scores.append(classify_data(ground_truth[mask], guess[mask]))

########### BIRKHOF ############################################################
print("Loading birkhoff...")

times = []
lyapunov_b_data = []

for key in f_birkhoff.keys():
    if not key.isdigit():
        continue
    times.append(int(key))
    lyapunov_b_data.append(f_birkhoff[key]["lyapunov_b_x"][:])

# sort times and lyapunov_b data
times, lyapunov_b_data = zip(*sorted(zip(times, lyapunov_b_data)))

# convert to numpy arrays
times = np.array(times)
lyapunov_b_data = np.array(lyapunov_b_data)

lyapunov_b_thresholds = []
lyapunov_b_post_data = []
lyapunov_b_guesses = []
lyapunov_b_scores = []
for t, data in zip(times, lyapunov_b_data):
    data[mask & (np.isnan(data))] = np.nanmax(data[mask])
    data = np.log10(data)
    data[mask & (np.isnan(data))] = np.nanmax(data[mask])
    data[mask & (np.isinf(data))] = np.nanmax(data[mask & (~np.isinf(data))])
    data[~mask] = np.nan
    lyapunov_b_post_data.append(data)
    lyapunov_b_thresholds.append(find_threshold_smart(data[mask], where_chaos="higher"))
    guess = data > lyapunov_b_thresholds[-1]
    lyapunov_b_guesses.append(guess)
    lyapunov_b_scores.append(classify_data(ground_truth[mask], guess[mask]))

########### NO BIRKHOF #########################################################

times = []
lyapunov_nob_data = []

for key in f_birkhoff.keys():
    if not key.isdigit():
        continue
    times.append(int(key))
    lyapunov_nob_data.append(f_birkhoff[key]["lyapunov_x"][:])

# sort times and lyapunov_nob data
times, lyapunov_nob_data = zip(*sorted(zip(times, lyapunov_nob_data)))

# convert to numpy arrays
times = np.array(times)
lyapunov_nob_data = np.array(lyapunov_nob_data)

lyapunov_nob_thresholds = []
lyapunov_nob_post_data = []
lyapunov_nob_guesses = []
lyapunov_nob_scores = []
for t, data in zip(times, lyapunov_nob_data):
    data[mask & (np.isnan(data))] = np.nanmax(data[mask])
    data = np.log10(data)
    data[mask & (np.isnan(data))] = np.nanmax(data[mask])
    data[mask & (np.isinf(data))] = np.nanmax(data[mask & (~np.isinf(data))])
    data[~mask] = np.nan
    lyapunov_nob_post_data.append(data)
    lyapunov_nob_thresholds.append(
        find_threshold_smart(data[mask], where_chaos="higher")
    )
    guess = data > lyapunov_nob_thresholds[-1]
    lyapunov_nob_guesses.append(guess)
    lyapunov_nob_scores.append(classify_data(ground_truth[mask], guess[mask]))


########### REM ################################################################
print("Loading rem...")

times = []
rem_data = []

for key in f_rem.keys():
    if key == "0":
        continue
    # if key is not the string of an integer, skip
    if not key.isdigit():
        continue
    times.append(int(key))
    rem_data.append(
        np.sqrt(
            (f_rem["0"]["x"][:] - f_rem[key]["x"][:]) ** 2
            + (f_rem["0"]["px"][:] - f_rem[key]["px"][:]) ** 2
            + (f_rem["0"]["y"][:] - f_rem[key]["y"][:]) ** 2
            + (f_rem["0"]["py"][:] - f_rem[key]["py"][:]) ** 2
        )
    )

# sort times and rem data
times, rem_data = zip(*sorted(zip(times, rem_data)))

# convert to numpy arrays
times = np.array(times)
rem_data = np.array(rem_data)

rem_thresholds = []
rem_best_thresholds = []
rem_post_data = []
rem_guesses = []
rem_scores = []
for t, data in zip(times, rem_data):
    data = np.log10(data)
    data[mask & (np.isnan(data))] = np.nanmax(data[mask])
    data[mask & (np.isinf(data))] = np.nanmax(data[mask & (~np.isinf(data))])
    data[~mask] = np.nan
    rem_post_data.append(data)
    rem_thresholds.append(find_threshold_smart(data[mask], where_chaos="higher"))
    rem_best_thresholds.append(find_best_threshold(ground_truth[mask], data[mask]))
    guess = data > rem_thresholds[-1]
    rem_guesses.append(guess)
    rem_scores.append(classify_data(ground_truth[mask], guess[mask]))


with open(f"{idx}_processed.pkl", "wb") as f:
    pickle.dump(
        {
            "times": times,
            "times_fma": times_fma,
            "initial_x": initial_x,
            "initial_px": initial_px,
            "initial_y": initial_y,
            "initial_py": initial_py,
            "ground_truth_data": ground_truth_data,
            "ground_truth": ground_truth,
            "mask": mask,
            "lyapunov_thresholds": lyapunov_thresholds,
            "lyapunov_post_data": lyapunov_post_data,
            "lyapunov_scores": lyapunov_scores,
            "megno_thresholds": megno_thresholds,
            "megno_post_data": megno_post_data,
            "megno_scores": megno_scores,
            "fli_x_thresholds": fli_x_thresholds,
            "fli_x_post_data": fli_x_post_data,
            "fli_x_scores": fli_x_scores,
            "gali_thresholds": gali_thresholds,
            "gali_post_data": gali_post_data,
            "gali_scores": gali_scores,
            "fma_birkhoff_thresholds": fma_birkhoff_thresholds,
            "fma_birkhoff_post_data": fma_birkhoff_post_data,
            "fma_birkhoff_scores": fma_birkhoff_scores,
            "fma_fft_thresholds": fma_fft_thresholds,
            "fma_fft_post_data": fma_fft_post_data,
            "fma_fft_scores": fma_fft_scores,
            "lyapunov_b_thresholds": lyapunov_b_thresholds,
            "lyapunov_b_post_data": lyapunov_b_post_data,
            "lyapunov_b_scores": lyapunov_b_scores,
            "lyapunov_nob_thresholds": lyapunov_nob_thresholds,
            "lyapunov_nob_post_data": lyapunov_nob_post_data,
            "lyapunov_nob_scores": lyapunov_nob_scores,
            "rem_thresholds": rem_thresholds,
            "rem_best_thresholds": rem_best_thresholds,
            "rem_post_data": rem_post_data,
            "rem_scores": rem_scores,
        },
        f,
    )
