"""Regression analysis pipeline for Hyser dataset (1DOF and MVC).

Loads preprocessed EMG features from pickle, trains MLP regressors to predict
force/stiffness from EMG features, evaluates with multiple metrics, and generates
annotated publication-ready plots.
"""

import pickle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from finger_impedance.core.functions import (
    butter_lowpass_filter,
    evaluate_regression_metrics,
    moving_average,
    mape,
    nrmse1,
    nrmse2,
    r_square,
    rmse,
    rmspe,
    vaf,
)

plt.style.use('bmh')
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["figure.figsize"] = (20, 10)

# --- Data loading ---

filename = 'regression_force_hyser_lin.sav'
with open('hyser/s01_hyser_1dof.pkl', 'rb') as handle:
    data = pickle.load(handle)
with open('hyser/s01_hyser_mvc.pkl', 'rb') as handle:
    data_mvc = pickle.load(handle)

ds = [data, data_mvc]
data_comb: Dict[str, np.ndarray] = {}
for k in data.keys():
    data_comb[k] = np.concatenate(list(d[k] for d in ds))

# --- Configuration ---

subject = 1
data_type = 'mvc'  # 'mvc' or '1dof'
to_train = 0  # if 1 algorithm will be trained
target = 's'  # 's' for stiffness, 'f' for force

if data_type == 'mvc':
    data = data_mvc
    label_list = ['Thumb\n flex', 'Thumb\n extend', 'Index\n flex', 'Index\n extend', 'Middle\n flex',
                  'Middle\n extend', 'Ring\n flex', 'Ring\nextend', 'Little\n flex', 'Little\n extend', 'Thumb_mvc',
                  'Index_mvc', 'Middle_mvc', 'Ring_mvc', 'Little_mvc']
    if target == 'f':
        filename = f'regression_force_hyser_mvc_s{subject}.sav'
    else:
        filename = f'regression_stiffness_hyser_mvc_s{subject}.sav'
else:
    label_list = ['Thumb', 'Index', 'Middle', 'Ring', 'Little', 'Thumb_mvc', 'Index_mvc', 'Middle_mvc', 'Ring_mvc',
                  'Little_mvc']
    if target == 'f':
        filename = f'regression_force_hyser_1dof_s{subject}.sav'
    else:
        filename = f'regression_stiffness_hyser_1dof_s{subject}.sav'

# --- Feature unpacking ---

feature_keys = ['rms_ext', 'mav_ext', 'iav_ext', 'var_ext', 'wl_ext', 'mf_ext',
                'pf_ext', 'mp_ext', 'tp_ext', 'sm_ext', 'msf_ext', 'movement_id',
                'force', 'stiffness']
(rms_ext, mav_ext, iav_ext, var_ext, wl_ext, mf_ext, pf_ext, mp_ext, tp_ext,
 sm_ext, msf_ext, movement_id, force, stiffness_estimation) = list(map(data.get, feature_keys))

movement_id = np.round([x - 1 for x in movement_id], 0)
normalized_stiffness = stiffness_estimation
moav_stiffness = np.empty((normalized_stiffness.shape[0], normalized_stiffness.shape[1]))

index = 0
for i in stiffness_estimation.T:
    moav_stiffness[:, index] = moving_average(i, 20)
    index += 1

# Normalize stiffness to 0-100%
shape1, shape2 = moav_stiffness.shape
moav_stiffness_flat = moav_stiffness.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 100))
scaler.fit(moav_stiffness_flat)
normalized_stiffness = scaler.transform(moav_stiffness_flat).reshape(shape1, shape2)

# Normalize force to -100 to 100%
shape1, shape2 = force.shape
force_flat = force.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(-100, 100))
scaler.fit(force_flat)
scaled_force = scaler.transform(force_flat).reshape(shape1, shape2)
normalized_force = scaled_force
force = force_flat.reshape(shape1, shape2)

# --- Plot configuration ---

# Alternating colors for movement annotation spans
color_list = ['black', 'yellow'] * 40

force_labels = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
finger_colors = ['teal', 'black', 'blue', 'red', 'grey', 'orange']

# --- Stiffness and force overview plot ---

finger_1 = 0
finger_2 = 1
ranges: Dict[float, np.ndarray] = {}
means: Dict[float, np.ndarray] = {}

fig2, ax2 = plt.subplots(2, 1)
for k in range(finger_1, finger_2):
    ax2[0].plot(normalized_stiffness[:, k], label=force_labels[k], color=finger_colors[k])
ax2[0].set_ylabel('Estimated normalized stiffness (%)', fontsize=15)
ax2[0].set_yticks(np.arange(0, 110, 10), ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
ax2[0].set_ylim(0, 110)

for k in range(finger_1, finger_2):
    ax2[1].plot(normalized_force[:, k], label=force_labels[k], color=finger_colors[k])
ax2[1].set_xlabel('time (epoch)', fontsize=15)
ax2[1].set_ylabel('Force percentage (%)', fontsize=15)

for i in np.unique(movement_id):
    ranges[i] = np.where(movement_id == int(i))
    data_range = np.r_[ranges[i][0]]
    means[i] = np.round(np.mean(normalized_stiffness[data_range], axis=0), 2)
    color_idx = np.where(np.unique(movement_id) == i)[0][0]
    ax2[0].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1,
                   color=color_list[color_idx], label=label_list[int(i)])
    ax2[0].annotate(label_list[int(i)], xy=(int(ranges[i][0][0]), 100), fontsize=15)
    ax2[1].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1,
                   color=color_list[color_idx])

df = pd.DataFrame.from_dict(means).T
df.to_excel('means2.xlsx')

ax2[1].legend(loc='lower right', fontsize=15)
ax2[1].set_ylim(-100, 110)
print(means)

# --- Feature combination and regression ---

tdf4 = np.hstack((rms_ext, wl_ext))
fd2 = np.hstack((tp_ext, sm_ext))
tfdf = np.hstack((tdf4, fd2))

X = tfdf
if target == 'f':
    y = force
else:
    y = normalized_stiffness

reg: Dict[int, MLPRegressor] = {}
predicted_force: Dict[int, np.ndarray] = {}
score = pd.DataFrame()

if to_train == 1:
    for i in range(5):
        print(f'Regression for {i + 1}')
        reg[i] = MLPRegressor(activation='tanh', random_state=1, max_iter=500).fit(X, y[:, i])
    pickle.dump(reg, open(filename, 'wb'))
else:
    reg = pickle.load(open(filename, 'rb'))

for i in range(5):
    predicted_force[i] = reg[i].predict(X)

# Apply moving average smoothing
moav_predicted = np.empty((predicted_force[1].shape[0], 6))
moav_y = np.empty((predicted_force[1].shape[0], 6))
index = 0

for i in range(5):
    moav_predicted[:, index] = moving_average(predicted_force[i][:], 20)
    index += 1
predicted_force_arr = moav_predicted.T

index = 0
for i in y.T:
    moav_y[:, index] = moving_average(i, 20)
    index += 1
force = moav_y
y = moav_y

score_rows = []
for i in range(5):
    score_rows.append(evaluate_regression_metrics(predicted_force_arr[i], y[:, i], force_labels[i]))
score = pd.concat(score_rows)

score.to_excel('regression_' + target + data_type + '.xlsx')
print(score)

# --- Prediction vs ground truth plot ---

fig2, ax2 = plt.subplots(5)
ax2[0].plot(predicted_force_arr[1], '--', label='index estimated')
ax2[0].plot(force[:, 1], label='index experimental')
ax2[1].plot(predicted_force_arr[2], '--', label='middle estimated')
ax2[1].plot(force[:, 2], label='middle experimental')
ax2[2].plot(predicted_force_arr[3], '--', label='ring estimated')
ax2[2].plot(force[:, 3], label='ring experimental')
ax2[3].plot(predicted_force_arr[4], '--', label='little estimated')
ax2[3].plot(force[:, 4], label='little experimental')
ax2[4].plot(predicted_force_arr[0], '--', label='thumb estimated')
ax2[4].plot(force[:, 0], label='thumb experimental')
ax2[-1].set_xlabel('time')

for ax in ax2:
    ax.legend(loc=7)

for i in np.unique(movement_id):
    ranges[i] = np.where(movement_id == int(i))
    color_idx = np.where(np.unique(movement_id) == i)[0][0]
    ax2[0].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1,
                   color=color_list[color_idx], label=label_list[int(i)])
    ax2[0].annotate(label_list[int(i)], xy=(int(ranges[i][0][0]), 0.8), fontsize=10)
    for j in range(1, 5):
        ax2[j].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1,
                       color=color_list[color_idx])

if target == 'f':
    fig2.text(0.09, 0.5, 'Force (%MVC)', va='center', rotation='vertical', fontsize=14)
else:
    fig2.text(0.09, 0.5, 'Stiffness (%)', va='center', rotation='vertical', fontsize=14)

figname = 'fig/est_' + target + '_' + data_type + '.pdf'
fig2.savefig(figname, format='pdf')

plt.show()
