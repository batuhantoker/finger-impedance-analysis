"""Multi-subject stiffness and force averaging/visualization for Hyser dataset.

Loads per-subject pickle files, normalizes stiffness/force with MinMaxScaler,
computes cross-subject mean/std, and plots with confidence bands and movement
annotations.
"""

import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from finger_impedance.core.functions import moving_average


def rms(data: np.ndarray, epoch: int) -> np.ndarray:
    """Compute epoch-wise mean of multi-channel data.

    Args:
        data: 2D array of shape (n_samples, n_channels).
        epoch: Number of samples per epoch window.

    Returns:
        Array of shape (n_segments, n_channels) with mean per epoch.
    """
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0:number_of_segments * epoch, :], number_of_segments)
    RMS = np.empty([number_of_segments, data.shape[1]])
    for i in range(number_of_segments):
        RMS[i, :] = np.mean(splitted_data[i], axis=0)
    return RMS


if __name__ == "__main__":
    plt.style.use('bmh')
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["figure.figsize"] = (10, 20)
    plt.rcParams.update({'font.size': 12})
    with open('s1_hyser.pkl', 'rb') as handle:
        data1 = pickle.load(handle)
    with open('s2_hyser.pkl', 'rb') as handle:
        data2 = pickle.load(handle)
    with open('s3_hyser.pkl', 'rb') as handle:
        data3 = pickle.load(handle)
    with open('s4_hyser.pkl', 'rb') as handle:
        data4 = pickle.load(handle)
    data_list = [data1, data2, data3, data4]
    data_length = [
        data1['stiffness'].shape[0],
        data2['stiffness'].shape[0],
        data3['stiffness'].shape[0],
        data4['stiffness'].shape[0],
    ]
    normalized_stiffness = np.zeros((4, min(data_length), data2['stiffness'].shape[1]))
    normalized_force = np.zeros((4, min(data_length), data2['stiffness'].shape[1]))

    k = 0
    for i in data_list:
        (
            rms_ext, mav_ext, iav_ext, var_ext, wl_ext, mf_ext, pf_ext, mp_ext,
            tp_ext, sm_ext, msf_ext, movement_id, force, stiffness_estimation,
        ) = list(map(i.get, [
            'rms_ext', 'mav_ext', 'iav_ext', 'var_ext', 'wl_ext', 'mf_ext',
            'pf_ext', 'mp_ext', 'tp_ext', 'sm_ext', 'msf_ext',
            'movement_id', 'force', 'stiffness',
        ]))
        normalized_stiffness[k, :, :] = stiffness_estimation[0:min(data_length), :]
        movement_id = np.round([x - 1 for x in movement_id], 0)
        shape1, shape2 = force[0:min(data_length), :].shape
        force = force[0:min(data_length), :].reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-100, 100))
        scaler.fit(force)
        normalized_force[k, :, :] = scaler.transform(force).reshape(shape1, shape2)
        k = k + 1

    moav_stiffness = np.zeros_like(normalized_stiffness)

    for i in range(0, 4):
        index = 0
        moav_stiffness2 = np.empty(
            (normalized_stiffness.shape[1], normalized_stiffness.shape[2])
        )
        for k in normalized_stiffness[i, :, :].T:
            moav_stiffness2[:, index] = np.array(moving_average(k, 20))
            index = index + 1
        shape1, shape2 = moav_stiffness2.shape
        moav_stiffness2 = moav_stiffness2.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaler.fit(moav_stiffness2)
        moav_stiffness[i] = scaler.transform(moav_stiffness2).reshape(shape1, shape2)

    stiffness_mean = np.mean(moav_stiffness, axis=0)
    stiffness_std = np.std(moav_stiffness, axis=0)
    print(np.mean(stiffness_std))
    mean_plus_std = stiffness_mean + stiffness_std
    mean_min_std = stiffness_mean - stiffness_std
    force_mean = np.mean(normalized_force, axis=0)
    force_std = np.std(normalized_force, axis=0)
    mean_plus_std2 = force_mean + force_std
    mean_min_std2 = force_mean - force_std
    movement_id = movement_id[0:min(data_length)]

    ranges = {}
    color_list = ['black', 'yellow'] * 40
    label_list = ['Thumb f', 'Thumb e', 'Index f', 'Index e', 'Middle f',
                  'Middle e', 'Ring f', 'Ring e', 'Little f', 'Little e']
    label_list = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    force_labels = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

    finger_1 = 4

    finger_2 = 2
    fig2, ax2 = plt.subplots(2, 1)
    ax2[0].plot(stiffness_mean[:, finger_1], label=force_labels[finger_1])
    ax2[0].plot(mean_plus_std[:, finger_1], 'r', linewidth=1, label=r'mean$\pm$std')
    ax2[0].plot(mean_min_std[:, finger_1], 'r', linewidth=1)
    ax2[0].fill_between(
        np.arange(0, min(data_length), 1),
        mean_plus_std[:, finger_1],
        mean_min_std[:, finger_1],
        alpha=0.1,
        color='red',
    )
    ax2[0].set_ylabel('Estimated normalized stiffness (%)')
    ax2[0].set_xlabel('time (epoch)')
    ax2[1].plot(force_mean[:, finger_1], label=force_labels[finger_1])
    ax2[1].plot(mean_plus_std2[:, finger_1], 'r', linewidth=1, label=r'mean$\pm$std')
    ax2[1].plot(mean_min_std2[:, finger_1], 'r', linewidth=1)
    ax2[1].fill_between(
        np.arange(0, min(data_length), 1),
        mean_plus_std2[:, finger_1],
        mean_min_std2[:, finger_1],
        alpha=0.1,
        color='red',
    )
    ax2[1].set_xlabel('time (epoch)')
    ax2[1].set_ylabel('Force percentage (%)')
    for i in np.unique(movement_id):
        ranges[i] = np.where(movement_id == int(i))
        ax2[0].axvspan(
            int(ranges[i][0][0]), int(ranges[i][0][-1]),
            alpha=0.1,
            color=color_list[np.where(np.unique(movement_id) == i)[0][0]],
            label=label_list[int(i)],
        )
        ax2[0].annotate(label_list[int(i)], xy=(int(ranges[i][0][0]), 100), fontsize=15)
        ax2[1].axvspan(
            int(ranges[i][0][0]), int(ranges[i][0][-1]),
            alpha=0.1,
            color=color_list[np.where(np.unique(movement_id) == i)[0][0]],
        )
    box = ax2[1].get_position()
    ax2[1].legend(loc='lower left', fontsize=15)
    ax2[0].set_ylim(0, 100)
    fig2.suptitle('Estimated stiffness and Force for Subjects 1,2,3,4')
    figname = 'fig/avg_' + f'{force_labels[finger_1]}' + '_' + '2.pdf'
    fig2.savefig(figname, format='pdf')
    plt.show()
