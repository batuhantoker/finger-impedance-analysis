"""Batch stiffness/force analysis across all Malesevic dataset subjects.

Loads per-subject pickle files, normalizes stiffness/force, computes cross-subject
mean/std/max statistics, exports to Excel, and generates publication plots.
"""

import pickle
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from finger_impedance.core.functions import moving_average, pct_change

plt.style.use("bmh")
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["figure.figsize"] = (20, 10)


finger_1 = 3
finger_2 = finger_1 + 1
stiffness = []
forces = []
ranges = {}
means = np.empty((20, 6, 13))
maxs = np.empty((20, 6, 13))
stds = np.empty((20, 6, 13))

subject_range = chain(range(1, 5), range(6, 21))

for i in subject_range:
    subject = i

    with open(f"males/data_s{subject}.pkl", "rb") as handle:
        data = pickle.load(handle)
    force, movement_id, stiffness_estimation = list(
        map(data.get, ["force", "movement_id", "stiffness"])
    )
    normalized_stiffness = stiffness_estimation
    moav_stiffness = np.empty(
        (normalized_stiffness.shape[0], normalized_stiffness.shape)[1]
    )

    index = 0
    for z in stiffness_estimation.T:
        moav_stiffness[:, index] = moving_average(z, 20)
        index = index + 1

    shape1, shape2 = moav_stiffness.shape
    moav_stiffness = moav_stiffness.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(moav_stiffness)
    normalized_stiffness = scaler.transform(moav_stiffness).reshape(shape1, shape2)

    force = force.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-100, 100))
    scaler.fit(force)
    scaled_force = scaler.transform(force).reshape(shape1, shape2)
    normalized_force = scaled_force
    force = force.reshape(shape1, shape2)
    for k in np.unique(movement_id):
        ranges[k] = np.where(movement_id == int(k))
        data_range = np.r_[ranges[k][0]]
        means[subject - 1, :, k.astype(int)] = np.round(
            np.mean(normalized_stiffness[data_range, 0:6], axis=0), 2
        )
        maxs[subject - 1, :, k.astype(int)] = np.round(
            np.max(normalized_stiffness[data_range, 0:6], axis=0), 2
        )
        stds[subject - 1, :, k.astype(int)] = np.round(
            np.std(normalized_stiffness[data_range, 0:6], axis=0), 2
        )
    stiffness.append(normalized_stiffness)
    forces.append(force)

means = np.delete(means, 4, 0)  # remove 5th subject
maxs = np.delete(maxs, 4, 0)  # remove 5th subject
stds = np.delete(stds, 4, 0)  # remove 5th subject
data_length = min([x.shape[0] for x in stiffness])

stiffness = np.array([x[0:data_length, :] for x in stiffness])
forces = np.array([x[0:data_length, :] for x in forces])

force_labels = ["Index", "Middle", "Ring", "Little", "Thumb left-right", "Thumb up-down", "thumb accumulated"]
finger_colors = ["black", "blue", "red", "grey", "teal", "sienna", "teal"]
color_list = (["red", "black", "yellow"] * 30)[:90]
label_list = [
    "rest", "Little\n flex", "Little\n extend", "Ring\n flex", "Ring\n extend",
    "Middle\n flex", "Middle\n extend", "Index\n flex", "Index\n extend",
    "Thumb:\n down", "Thumb:\n up", "Thumb:\n left", "Thumb:\n right",
    "Wrist: bend", "Wrist: rotate anti-clockwise", "Wrist: rotate clockwise",
    "Little finger: bend+Ring finger: bend", "Little finger: bend+Thumb: down",
    "Little finger: bend+Thumb: left", "Little finger: bend+thumb: right",
    "Little finger: bend+wrist: bend", "Little finger: bend+Wrist: stretch",
    "Little finger: bend+Wrist: rotate anti-clockwise",
    "Little finger: bend+Wrist: rotate clockwise",
    "Ring finger: bend+Middle finger: bend", "Ring finger: bend+Thumb: down",
    "Ring finger: bend+Thumb: left", "Ring finger: bend+Thumb: right",
    "Ring finger: bend+Wrist: bend", "Ring finger: bend+Wrist: stretch",
    "Ring finger: bend+Wrist: rotate anti-clockwise",
    "Ring finger: bend+Wrist: rotate clockwise",
    "Middle finger: bend+Index finger: bend", "Middle finger: bend+Thumb: down",
    "Middle finger: bend+Thumb: left", "Middle finger: bend+Thumb: right",
    "Middle finger: bend+Wrist: bend", "Middle finger: bend+Wrist: stretch",
    "Middle finger: bend+Wrist: rotate anti-clockwise",
    "Middle finger: bend+Wrist: rotate clockwise",
    "Index finger: bend+Thumb: down", "Index finger: bend+Thumb: left",
    "Index finger: bend+Thumb: right", "Index finger: bend+Wrist: bend",
    "Index finger: bend+Wrist: stretch",
    "Index finger: bend+Wrist: rotate anti-clockwise",
    "Index finger: bend+Wrist: rotate clockwise",
    "Thumb: down+Thumb: left", "Thumb: down+Thumb: right",
    "Thumb: down+Thumb:bend", "Thumb: down+Thumb:stretch",
    "Thumb: down+Wrist: rotate anti-clockwise",
    "Thumb: down+Wrist: rotate clockwise",
    "Wrist: bend+Wrist: rotate anti-clockwise",
    "Wrist: bend+Wrist: rotate clockwise",
    "Wrist: stretch+Wrist: rotate anti-clockwise",
    "Wrist: stretch+Wrist: rotate clockwise",
    "Extend all fingers (without thumb)", "All fingers: bend (without thumb)",
    "Extend all fingers (without thumb)", "Palmar grasp",
    "Wrist: rotate anti-clockwise with the Palmar grasp",
    "Pointing (index: stretch, all other: bend)", "3-digit pinch",
    "3-digit pinch with Wrist: anti-clockwise rotation",
    "Key grasp with Wrist: anti-clockwise rotation", "",
]

stiffness_mean = np.mean(stiffness, axis=0)

data_length = stiffness_mean.shape[0]
stiffness_std = np.std(stiffness, axis=0)
mean_plus_std = stiffness_mean + stiffness_std
mean_min_std = stiffness_mean - stiffness_std
force_mean = np.mean(forces, axis=0)
force_std = np.std(forces, axis=0)
mean_plus_std2 = force_mean + force_std
mean_min_std2 = force_mean - force_std
movement_id = movement_id[0:data_length]

fig2, ax2 = plt.subplots(2, 1)
for k in range(finger_1, finger_2):
    ax2[0].plot(stiffness_mean[:, k], label=force_labels[k], color=finger_colors[k], linewidth=2)
    ax2[0].plot(mean_plus_std[:, k], color=finger_colors[k], linewidth=0.5, label=r"mean$\pm$std")
    ax2[0].plot(mean_min_std[:, k], color=finger_colors[k], linewidth=0.5)
    ax2[0].fill_between(
        np.arange(0, data_length, 1), mean_plus_std[:, k], mean_min_std[:, k],
        alpha=0.1, color=finger_colors[k],
    )
    ax2[0].set_ylabel("Estimated normalized stiffness [%]")
    ax2[1].plot(force_mean[:, k], label=force_labels[k], linewidth=2)
    ax2[1].plot(mean_plus_std2[:, k], color=finger_colors[k], linewidth=0.5, label=r"mean$\pm$std")
    ax2[1].plot(mean_min_std2[:, k], color=finger_colors[k], linewidth=0.5)
    ax2[1].fill_between(
        np.arange(0, data_length, 1), mean_plus_std2[:, k], mean_min_std2[:, k],
        alpha=0.1, color=finger_colors[k],
    )
    ax2[1].set_xlabel("time [epoch]")
    ax2[1].set_ylabel("Force [N]")
ranges = {}

for i in np.unique(movement_id):
    ranges[i] = np.where(movement_id == int(i))
    data_range = np.r_[ranges[i][0]]
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

ax2[1].legend(loc="lower left", fontsize=15)
ax2[0].set_ylim(0, 100)
fig2.suptitle(
    f"{force_labels[finger_1]} finger extracted stiffness from force and force signals for all subjects, dataset 1",
    fontsize=15,
)

# Means and maxs
df_mean = pd.DataFrame(np.mean(means, axis=0))
df_max = pd.DataFrame(np.mean(maxs, axis=0))
df_std = pd.DataFrame(np.mean(stds, axis=0))
df_std.columns = label_list[0:13]
df_std.index = force_labels[0:6]
df_mean.columns = label_list[0:13]
df_mean.index = force_labels[0:6]
df_max.columns = label_list[0:13]
df_max.index = force_labels[0:6]
df_mean.to_excel("malesevic_means.xlsx")
df_max.to_excel("malesevic_maxs.xlsx")
df_std.to_excel("malesevic_stds.xlsx")
print(df_mean.columns)
ax5 = df_mean.T.plot(kind="line", title="Mean of the estimated stiffness values for dataset 1", color=finger_colors)
pct = pct_change(df_mean.T) * 100
pct.plot(kind="bar", title="Mean of the estimated stiffness values for dataset 1", color=finger_colors, ax=ax5)
ax5.set_ylabel("Percentage [%]")
ax4 = df_max.T.plot(kind="line", title="Max of the estimated stiffness values for dataset 1", color=finger_colors)
pct = pct_change(df_max.T) * 100
pct.plot(kind="bar", color=finger_colors, ax=ax4)
ax4.set_ylabel("Percentage [%]")
ax4.set_xlabel("Movement")
ax3 = df_std.T[[force_labels[finger_1]]].plot(
    kind="bar",
    title="Mean standard deviation of the estimated stiffness values for dataset 1",
    color=finger_colors[finger_1],
)
ax3.set_ylabel("Percentage [%]")
ax3.set_xlabel("Movement")
plt.show()
