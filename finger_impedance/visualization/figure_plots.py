"""Regression analysis pipeline for Malesevic dataset.

Loads preprocessed EMG features from pickle, trains MLP regressors to predict
force/stiffness from EMG features, evaluates with multiple metrics, and generates
publication-ready plots with movement annotations and per-finger comparisons.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from finger_impedance.core.functions import (
    evaluate_regression_metrics,
    moving_average,
    pct_change,
)

if __name__ == "__main__":
    plt.style.use('bmh')
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["figure.figsize"] = (20, 10)
    with open('males/data_s1.pkl', 'rb') as handle:
        data = pickle.load(handle)

    target = 'f'  # s for stiffness f force
    to_train = 0  # if 1 algorithm will be trained
    if target == 'f':
        filename = 'regression_force.sav'
    else:
        filename = 'regression_stiffness.sav'

    (
        rms_flex, mav_flex, iav_flex, var_flex, wl_flex, mf_flex, pf_flex, mp_flex,
        tp_flex, sm_flex, msf_flex, rms_ext, mav_ext, iav_ext, var_ext, wl_ext,
        mf_ext, pf_ext, mp_ext, tp_ext, sm_ext, msf_ext, movement_id, force,
        stiffness_estimation,
    ) = list(map(data.get, [
        'rms_flex', 'mav_flex', 'iav_flex', 'var_flex', 'wl_flex', 'mf_flex',
        'pf_flex', 'mp_flex', 'tp_flex', 'sm_flex', 'msf_flex',
        'rms_ext', 'mav_ext', 'iav_ext', 'var_ext', 'wl_ext', 'mf_ext',
        'pf_ext', 'mp_ext', 'tp_ext', 'sm_ext', 'msf_ext',
        'movement_id', 'force', 'stiffness',
    ]))
    normalized_stiffness = stiffness_estimation
    moav_stiffness = np.empty((normalized_stiffness.shape[0], normalized_stiffness.shape[1]))
    index = 0
    for i in stiffness_estimation.T:
        moav_stiffness[:, index] = moving_average(i, 20)
        index = index + 1

    shape1, shape2 = moav_stiffness.shape
    moav_stiffness = moav_stiffness.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(moav_stiffness)
    normalized_stiffness = scaler.transform(moav_stiffness).reshape(shape1, shape2)
    normalized_stiffness[:, 7] = np.sqrt(
        normalized_stiffness[:, 6] ** 2 + normalized_stiffness[:, 5] ** 2
    )

    shape1, shape2 = force.shape
    force = force.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-100, 100))
    scaler.fit(force)
    scaled_force = scaler.transform(force).reshape(shape1, shape2)
    normalized_force = scaled_force
    force = force.reshape(shape1, shape2)
    max_ind_flex = []
    max_ind_ext = []
    for i in rms_flex:
        max_ind_flex.append(np.argsort(i)[::-1][:10])
    for i in rms_ext:
        max_ind_ext.append(np.argsort(i)[::-1][:10])

    ind_flex = np.array(max_ind_flex)
    ind_ext = np.array(max_ind_flex)
    feat_ind = np.hstack((ind_flex, ind_ext))

    ranges = {}
    means = {}
    label_list = [
        'rest', 'Little\n finger:\n flex', 'Little\nfinger:\n extend',
        'Ring\nfinger:\n flex', 'Ring\nfinger:\n extend',
        'Middle\nfinger:\n flex', 'Middle\nfinger:\n extend',
        'Index\nfinger:\n flex', 'Index\nfinger:\n extend',
        'Thumb:\n down', 'Thumb:\n up', 'Thumb:\n left', 'Thumb:\n right',
        'Wrist: bend', 'Wrist: rotate anti-clockwise', 'Wrist: rotate clockwise',
        'Little finger: bend+Ring finger: bend',
        'Little finger: bend+Thumb: down', 'Little finger: bend+Thumb: left',
        'Little finger: bend+thumb: right', 'Little finger: bend+wrist: bend',
        'Little finger: bend+Wrist: stretch',
        'Little finger: bend+Wrist: rotate anti-clockwise',
        'Little finger: bend+Wrist: rotate clockwise',
        'Ring finger: bend+Middle finger: bend',
        'Ring finger: bend+Thumb: down', 'Ring finger: bend+Thumb: left',
        'Ring finger: bend+Thumb: right',
        'Ring finger: bend+Wrist: bend', 'Ring finger: bend+Wrist: stretch',
        'Ring finger: bend+Wrist: rotate anti-clockwise',
        'Ring finger: bend+Wrist: rotate clockwise',
        'Middle finger: bend+Index finger: bend',
        'Middle finger: bend+Thumb: down', 'Middle finger: bend+Thumb: left',
        'Middle finger: bend+Thumb: right',
        'Middle finger: bend+Wrist: bend', 'Middle finger: bend+Wrist: stretch',
        'Middle finger: bend+Wrist: rotate anti-clockwise',
        'Middle finger: bend+Wrist: rotate clockwise',
        'Index finger: bend+Thumb: down', 'Index finger: bend+Thumb: left',
        'Index finger: bend+Thumb: right',
        'Index finger: bend+Wrist: bend', 'Index finger: bend+Wrist: stretch',
        'Index finger: bend+Wrist: rotate anti-clockwise',
        'Index finger: bend+Wrist: rotate clockwise',
        'Thumb: down+Thumb: left', 'Thumb: down+Thumb: right',
        'Thumb: down+Thumb:bend', 'Thumb: down+Thumb:stretch',
        'Thumb: down+Wrist: rotate anti-clockwise',
        'Thumb: down+Wrist: rotate clockwise',
        'Wrist: bend+Wrist: rotate anti-clockwise',
        'Wrist: bend+Wrist: rotate clockwise',
        'Wrist: stretch+Wrist: rotate anti-clockwise',
        'Wrist: stretch+Wrist: rotate clockwise',
        'Extend all fingers (without thumb)',
        'All fingers: bend (without thumb)',
        'Extend all fingers (without thumb)', 'Palmar grasp',
        'Wrist: rotate anti-clockwise with the Palmar grasp',
        'Pointing (index: stretch, all other: bend)',
        '3-digit pinch', '3-digit pinch with Wrist: anti-clockwise rotation',
        'Key grasp with Wrist: anti-clockwise rotation', '',
    ]

    force_labels = ['index', 'middle', 'ring', 'little', 'thumb left-right', 'thumb up-down', 'thumb accumulated']
    finger_colors = ['black', 'blue', 'red', 'grey', 'teal', 'sienna', 'teal']
    color_list = ['red', 'black', 'yellow'] * 30

    finger_1 = 1
    finger_2 = 2
    fig2, ax2 = plt.subplots(2, 1)
    for k in range(finger_1, finger_2):
        ax2[0].plot(normalized_stiffness[:, k], label=force_labels[k], color=finger_colors[k])
    ax2[0].set_ylabel('Estimated normalized stiffness (%)', fontsize=15)
    ax2[0].set_yticks(
        np.arange(0, 110, 10),
        ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],
    )
    ax2[0].set_ylim(0, 110)

    for k in range(finger_1, finger_2):
        ax2[1].plot(normalized_force[:, k], label=force_labels[k], color=finger_colors[k])
    ax2[1].set_xlabel('time (epoch)', fontsize=15)
    ax2[1].set_ylabel('Force percentage (%)', fontsize=15)
    for i in np.unique(movement_id):
        ranges[i] = np.where(movement_id == int(i))
        data_range = np.r_[ranges[i][0]]
        means[i] = np.round(np.mean(normalized_stiffness[data_range], axis=0), 2)
        ax2[0].axvspan(
            int(ranges[i][0][0]), int(ranges[i][0][-1]),
            alpha=0.1,
            color=color_list[np.where(np.unique(movement_id) == i)[0][0]],
            label=label_list[int(i)],
        )
        ax2[1].axvspan(
            int(ranges[i][0][0]), int(ranges[i][0][-1]),
            alpha=0.1,
            color=color_list[np.where(np.unique(movement_id) == i)[0][0]],
        )
        ax2[0].annotate(label_list[int(i)], xy=(int(ranges[i][0][0]), 100), fontsize=15)
    df = pd.DataFrame.from_dict(means).T
    df.to_excel('means1.xlsx')

    ax2[1].legend(loc='lower left', fontsize=15)
    ax2[1].set_ylim(-100, 110)

    rms_features = np.hstack((rms_flex, rms_ext))
    wl_features = np.hstack((wl_flex, wl_ext))
    tdf4 = np.hstack((rms_features, wl_features))
    tp_features = np.hstack((tp_flex, tp_ext))
    sm_features = np.hstack((sm_flex, sm_ext))
    fd2 = np.hstack((tp_features, sm_features))
    tfdf = np.hstack((fd2, tdf4))

    X = tfdf
    if target == 'f':
        y = force[:, 0:6]
    else:
        y = normalized_stiffness[:, 0:6]

    reg = {}
    predicted_force = {}
    score = pd.DataFrame()
    if to_train == 1:
        for i in range(6):
            print(f'Regression for {i + 1}')
            reg[i] = MLPRegressor(activation='tanh', random_state=1, max_iter=500).fit(X, y[:, i])
        pickle.dump(reg, open(filename, 'wb'))
    if to_train == 0:
        reg = pickle.load(open(filename, 'rb'))

    for i in range(6):
        predicted_force[i] = reg[i].predict(X)

    moav_predicted = np.empty((predicted_force[1].shape[0], 6))
    moav_y = np.empty((predicted_force[1].shape[0], 6))
    index = 0
    for i in range(6):
        moav_predicted[:, index] = moving_average(predicted_force[i][:], 20)
        index = index + 1
    predicted_force = moav_predicted.T
    index = 0
    for i in y.T:
        moav_y[:, index] = moving_average(i, 20)
        index = index + 1
    force = moav_y
    y = moav_y

    score_rows = []
    for i in range(6):
        score_rows.append(evaluate_regression_metrics(predicted_force[i], y[:, i], force_labels[i]))
    score = pd.concat(score_rows)

    score.to_excel('regression_' + target + '1.xlsx')
    print(score)
    fig2, ax2 = plt.subplots(6)
    ax2[0].plot(predicted_force[0], '--', label='index estimated')
    ax2[0].plot(force[:, 0], label='index experimental')
    ax2[1].plot(predicted_force[1], '--', label='middle estimated')
    ax2[1].plot(force[:, 1], label='middle experimental')
    ax2[2].plot(predicted_force[2], '--', label='ring estimated')
    ax2[2].plot(force[:, 2], label='ring experimental')
    ax2[3].plot(predicted_force[3], '--', label='little estimated')
    ax2[3].plot(force[:, 3], label='little experimental')
    ax2[4].plot(predicted_force[4], '--', label='thumb left/right estimated')
    ax2[4].plot(force[:, 4], label='thumb left/right experimental')
    ax2[-1].set_xlabel('time')
    ax2[5].plot(predicted_force[5], '--', label='thumb up/down estimated')
    ax2[5].plot(force[:, 5], label='thumb up/down experimental')

    for ax in ax2:
        ax.legend(loc=7)
    for i in np.unique(movement_id):
        ranges[i] = np.where(movement_id == int(i))
        ax2[0].axvspan(
            int(ranges[i][0][0]), int(ranges[i][0][-1]),
            alpha=0.1,
            color=color_list[np.where(np.unique(movement_id) == i)[0][0]],
            label=label_list[int(i)],
        )
        ax2[0].annotate(label_list[int(i)], xy=(int(ranges[i][0][0]), 6), fontsize=10)
        for j in range(1, 6):
            ax2[j].axvspan(
                int(ranges[i][0][0]), int(ranges[i][0][-1]),
                alpha=0.1,
                color=color_list[np.where(np.unique(movement_id) == i)[0][0]],
            )

    if target == 'f':
        fig2.text(0.09, 0.5, 'Force (N)', va='center', rotation='vertical', fontsize=14)
    else:
        fig2.text(0.09, 0.5, 'Stiffness (%)', va='center', rotation='vertical', fontsize=14)

    figname = 'fig/est_' + target + '_' + '1.pdf'
    fig2.savefig(figname, format='pdf')
    plt.show()
