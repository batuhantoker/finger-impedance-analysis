"""Data preprocessing and stiffness analysis pipeline for Hyser dataset (all subjects).

Reads per-subject pickle files, computes normalized stiffness across 20 subjects,
and generates mean/max/std statistics with publication-quality plots.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from finger_impedance.core.functions import moving_average, pct_change

try:
    import scienceplots  # noqa: F401
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use(['science', 'no-latex', 'grid'])
except ImportError:
    plt.rcParams.update(plt.rcParamsDefault)

finger_1=0
dataset='mvc'
stiffness=[]
forces=[]
ranges={}
means={}
maxs={}
stds={}

if dataset == 'mvc':
    label_list = ['Thumb\n flex', 'Thumb\n extend', 'Index\n flex', 'Index\n extend', 'Middle\n flex',
                  'Middle\n extend', 'Ring\n flex', 'Ring\nextend', 'Little\n flex', 'Little\n extend']
    data_number='3'
    means = np.empty((20, 5, 10))
    maxs = np.empty((20, 5, 10))
    stds = np.empty((20, 5, 10))
else:
    label_list = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    means = np.empty((20, 5, 5))
    maxs = np.empty((20, 5, 5))
    stds = np.empty((20, 5, 5))
    data_number = '2'
for i in range(1,21):
    subject="%02d" % i

    ## Import data
    filename='hyser/s'+subject+'_hyser_'+dataset+'.pkl'

    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    force,movement_id,stiffness_estimation = list(map(data.get,['force' ,'movement_id', 'stiffness']))
    movement_id = np.round([x - 1 for x in movement_id], 0)


    normalized_stiffness = stiffness_estimation  # normalize(stiffness_estimation, axis=0, norm='max')#
    moav_stiffness = np.empty((normalized_stiffness.shape[0], normalized_stiffness.shape)[1])

    index = 0
    for z in stiffness_estimation.T:
        moav_stiffness[:, index] = moving_average(z, 20)
        index = index + 1
    deleted_last_items=20
    moav_stiffness=moav_stiffness[:-deleted_last_items,:]
    force = force[:-deleted_last_items, :]
    movement_id = movement_id[:-deleted_last_items]

    shape1, shape2 = moav_stiffness.shape
    moav_stiffness = moav_stiffness.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(moav_stiffness)
    normalized_stiffness = scaler.transform(moav_stiffness).reshape(shape1, shape2)

    force = force.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-100, 100))
    scaler.fit(force)
    scaled_force = scaler.transform(force).reshape(shape1, shape2)
    normalized_force = scaled_force  # normalize(abs(force), axis=0, norm='max')
    force = force.reshape(shape1, shape2)  # scaled_force
    for k in np.unique(movement_id):
        ranges[k] = np.where(movement_id == int(k))
        data_range = np.r_[ranges[k][0]]
        means[int(i)-1,:,int(k)]=np.round(np.mean(normalized_stiffness[data_range], axis=0), 2)
        maxs[int(i)-1,:,int(k)]=np.round(np.max(normalized_stiffness[data_range], axis=0), 2)
        stds[int(i) - 1, :, int(k)] = np.round(np.std(normalized_stiffness[data_range], axis=0), 2)
    stiffness.append(normalized_stiffness)
    forces.append(force)
#means=np.delete(means,0,0) # remove 0th subject
#maxs=np.delete(maxs,0,0) # remove 0th subject
data_length=min([x.shape[0] for x in stiffness])

stiffness=np.array([x[0:data_length,:] for x in stiffness])#
forces = np.array([x[0:data_length,:] for x in forces])#
force_labels=['Thumb','Index','Middle','Ring','Little']

color_list=['black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow']

force_labels=['Thumb','Index','Middle','Ring','Little']

finger_colors = ['purple','blue','black','green','red','magenta']
finger_colors = ['teal','black','blue','red','grey','orange']

stiffness=np.array(stiffness)
data_length=stiffness.shape[1]
stiffness_mean=np.mean(stiffness, axis=0)
stiffness_std=np.std(stiffness, axis=0)
mean_plus_std = stiffness_mean+stiffness_std
mean_min_std = stiffness_mean-stiffness_std
force_mean=np.mean(forces, axis=0)
force_std=np.std(forces, axis=0)
mean_plus_std2 = force_mean+force_std
mean_min_std2 = force_mean-force_std
movement_id=movement_id[0:data_length]


fig2, ax2 = plt.subplots(2,1)
ax2[0].plot(stiffness_mean[:,finger_1],label=force_labels[finger_1], color = finger_colors[finger_1],linewidth=2)
ax2[0].plot(mean_plus_std[:,finger_1],color = finger_colors[finger_1], linewidth=0.5,label=r'mean$\pm$std')
ax2[0].plot(mean_min_std[:,finger_1],color = finger_colors[finger_1], linewidth=0.5)
ax2[0].fill_between(np.arange(0,data_length,1),mean_plus_std[:,finger_1], mean_min_std[:,finger_1], alpha = 0.1, color = finger_colors[finger_1])
ax2[0].set_ylabel('Estimated normalized stiffness [%]', fontsize=15)
#ax2[0].set_xlabel('time (epoch)')
ax2[1].plot(force_mean[:,finger_1],label=force_labels[finger_1], color = finger_colors[finger_1],linewidth=2)
ax2[1].plot(mean_plus_std2[:,finger_1],color = finger_colors[finger_1], linewidth=0.5,label=r'mean$\pm$std')
ax2[1].plot(mean_min_std2[:,finger_1],color = finger_colors[finger_1], linewidth=0.5)
ax2[1].fill_between(np.arange(0,data_length,1),mean_plus_std2[:,finger_1], mean_min_std2[:,finger_1], alpha = 0.1, color = finger_colors[finger_1])
ax2[1].set_xlabel('time [epoch]', fontsize=15)
ax2[1].set_ylabel('Force [% MVC] ', fontsize=15) #percentage (%)
for i in np.unique(movement_id):
    ranges[i] = np.where(movement_id == int(i))
    data_range=np.r_[ranges[i][0]]
    #means[i]= np.round(np.mean(normalized_stiffness[data_range],axis=0),2)
    #maxs[i] = np.round(np.max(normalized_stiffness[data_range], axis=0), 2)
    ax2[0].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1, color=color_list[np.where(np.unique(movement_id)==i)[0][0]],label=label_list[int(i)])
    ax2[0].annotate(label_list[int(i) ], xy=(int(ranges[i][0][0]), 100), fontsize=15)
    #ax2[1].annotate(label_list[int(i) ], xy=(int(ranges[i][0][0]), 100), fontsize=10)
    ax2[1].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1, color=color_list[np.where(np.unique(movement_id) == i)[0][0]])

ax2[1].legend(loc='lower left', fontsize=15)
ax2[0].set_ylim(0, 100)
fig2.suptitle(f'{force_labels[finger_1]} finger extracted stiffness from force and force signals for all subjects, dataset {data_number} - {dataset}', fontsize=15)

## Means and maxs
print(stds)
df_mean = pd.DataFrame(np.mean(means,axis=0))
df_max = pd.DataFrame(np.mean(maxs,axis=0)) # pd.DataFrame(maxs)#
df_std = pd.DataFrame(np.mean(stds,axis=0))
df_mean.columns = label_list
df_mean.index = force_labels
df_max.columns = label_list
df_max.index = force_labels
df_std.columns = label_list
df_std.index = force_labels
df_mean.to_excel('hyser_means'+dataset+'.xlsx')
df_max.to_excel('hyser_maxs'+dataset+'.xlsx')
df_std.to_excel('hyser_stds'+dataset+'.xlsx')
plt.plot() #[[force_labels[finger_1]]]
ax5=df_mean.T.plot(kind = "line",title = f"Mean of the estimated stiffness values for dataset {data_number} - {dataset}",color=finger_colors)
pct=pct_change(df_mean.T)*100
pct.plot(kind = "bar",color=finger_colors,ax=ax5) #[[force_labels[finger_1]]] [finger_1]
#ax5.legend([force_labels[finger_1],'Percentage change from thumb'])
ax5.set_ylabel("Percentage [%]",fontsize=15)
ax5.set_xlabel("Movement",fontsize=15)
ax4=df_max.T.plot(kind = "line",title = f"Max of the estimated stiffness values for dataset {data_number} - {dataset}",color=finger_colors) #[finger_1]
pct=pct_change(df_max.T)*100
pct.plot(kind = "bar",color=finger_colors,ax=ax4)
ax4.set_ylabel("Percentage [%]",fontsize=15)
ax4.set_xlabel("Movement",fontsize=15)
#ax4.legend([force_labels[finger_1],'Percentage change from thumb'])
ax3=df_std.T[[force_labels[finger_1]]].plot(kind = "bar",title = f"Mean standard deviation of the estimated stiffness values for dataset {data_number} - {dataset} ",color=finger_colors[finger_1])
ax3.set_ylabel("Percentage [%]",fontsize=15)
ax3.set_xlabel("Movement",fontsize=15)
plt.show()
