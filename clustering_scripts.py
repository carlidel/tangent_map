import numpy as np
import scipy as sp
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
