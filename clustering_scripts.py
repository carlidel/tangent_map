import matplotlib.patches as patches
import numpy as np
import scipy as sp
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def filter_data(mask, data, data_fix=True):
    if data_fix:
        data[np.isinf(data)] = np.nanmax(data)
        data[np.isnan(data)] = np.nanmax(data)
    data = data[mask]
    return data


def cover_extreme_outliers(data, m=10):
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)
    data_min = data_mean - m * data_std
    data_max = data_mean + m * data_std
    data_mask = np.logical_and(data >= data_min, data <= data_max)
    data[~data_mask] = np.nan
    return data


def find_threshold(data):
    data = data[~np.isnan(data) & ~np.isinf(data)]
    labels = KMeans(n_clusters=2, random_state=42).fit_predict(data.reshape(-1, 1))
    max_1 = np.max(data[labels == 0])
    max_2 = np.max(data[labels == 1])
    min_1 = np.min(data[labels == 0])
    min_2 = np.min(data[labels == 1])
    if max_1 > max_2:
        thresh = (max_2 + min_1) / 2
    else:
        thresh = (max_1 + min_2) / 2

    return thresh


def classify_data(ground_truth: np.ndarray, guess: np.ndarray):
    total = ground_truth.size
    true_positive = np.sum(ground_truth & guess)
    true_negative = np.sum(~ground_truth & ~guess)
    false_positive = np.sum(~ground_truth & guess)
    false_negative = np.sum(ground_truth & ~guess)

    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    return dict(
        total=total,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def get_extents(data_list, cover_m=10):
    val_min = np.nan
    val_max = np.nan
    for d in data_list:
        if cover_m is not None:
            d = cover_extreme_outliers(d, m=cover_m)
        val_min = np.nanmin([val_min, np.nanmin(d)])
        val_max = np.nanmax([val_max, np.nanmax(d)])
    return val_min, val_max


def compose_count_map(data_list, val_min, val_max, nbins=50):
    count_map = np.zeros((len(data_list), nbins))
    for i, d in enumerate(data_list):
        count, bins = np.histogram(
            d, bins=nbins, density=True, range=(val_min, val_max)
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        count_map[i, :] = count
    return count_map, bin_centers


def make_colormap(
    ax_colormap, count_map, data_list, time_list, val_min, val_max, patches_list=None
):
    mappable = ax_colormap.imshow(
        np.log10(count_map),
        aspect="auto",
        origin="lower",
        extent=(val_min, val_max, 0, len(data_list)),
    )
    ax_colormap.set_yticks(np.arange(len(time_list)) + 0.5)
    ax_colormap.set_yticklabels([f"$10^{int(np.log10(t))}$" for t in time_list])

    if patches_list is not None:
        d_rect_x = (val_max - val_min) * 0.005
        d_rect_y = 0.01
        for p in patches_list:
            rect = patches.Rectangle(
                (val_min + d_rect_x, p + d_rect_y),
                val_max - val_min - d_rect_x * 2,
                1 - d_rect_y * 2,
                linewidth=3,
                edgecolor="r",
                facecolor="none",
            )
            ax_colormap.add_patch(rect)

    return ax_colormap, mappable
