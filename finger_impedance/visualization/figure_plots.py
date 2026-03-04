import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score
from finger_impedance.core.functions import *
import pandas as pd

from scipy.signal import hilbert, find_peaks
from sklearn.utils import resample
plt.style.use('bmh')
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["figure.figsize"] = (20,10)
with open('males/data_s1.pkl', 'rb') as handle:
    data = pickle.load(handle)

target='f' # s for stiffness f force
to_train=0#if 1 algorithm will be trained
if target=='f':
    filename = 'regression_force.sav'
else:
    filename = 'regression_stiffness.sav'
###
rms_flex, mav_flex, iav_flex, var_flex, wl_flex, mf_flex, pf_flex, mp_flex, tp_flex, sm_flex, msf_flex,rms_ext, mav_ext, iav_ext, var_ext, wl_ext, mf_ext, pf_ext, mp_ext, tp_ext, sm_ext, msf_ext, movement_id, force, stiffness_estimation = list(map(data.get, ['rms_flex', 'mav_flex', 'iav_flex', 'var_flex', 'wl_flex', 'mf_flex', 'pf_flex', 'mp_flex', 'tp_flex', 'sm_flex', 'msf_flex','rms_ext', 'mav_ext', 'iav_ext', 'var_ext', 'wl_ext', 'mf_ext', 'pf_ext', 'mp_ext', 'tp_ext', 'sm_ext', 'msf_ext', 'movement_id','force','stiffness']) )
normalized_stiffness=stiffness_estimation#normalize(stiffness_estimation, axis=0, norm='max')#
moav_stiffness=np.empty((normalized_stiffness.shape[0],normalized_stiffness.shape)[1])
index=0
for i in stiffness_estimation.T:
    moav_stiffness[:,index]=moving_average(i,20)
    index=index+1


#scaler = MinMaxScaler(feature_range=(0, 100))
#scaler.fit(moav_stiffness)
#normalized_stiffness=scaler.transform(moav_stiffness)#normalize(moav_stiffness, axis=0, norm='l1')#normalized_stiffness#moav_stiffness#
shape1, shape2 = moav_stiffness.shape
moav_stiffness = moav_stiffness.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 100))
scaler.fit(moav_stiffness)
normalized_stiffness= scaler.transform(moav_stiffness).reshape(shape1, shape2)
normalized_stiffness[:,7]=np.sqrt(normalized_stiffness[:,6]**2+normalized_stiffness[:,5]**2)
#normalized_stiffness=np.array([i-np.mean(i) for i in normalized_stiffness])

shape1, shape2 = force.shape
force=force.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(-100, 100))
scaler.fit(force)
scaled_force=scaler.transform(force).reshape(shape1, shape2)
normalized_force=scaled_force #normalize(abs(force), axis=0, norm='max')
force=force.reshape(shape1, shape2)#scaled_force
max_ind_flex=[]
max_ind_ext=[]
for i in rms_flex:
    max_ind_flex.append(np.argsort(i)[::-1][:10])
for i in rms_ext:
    max_ind_ext.append(np.argsort(i)[::-1][:10])

ind_flex=np.array(max_ind_flex)
ind_ext=np.array(max_ind_flex)
feat_ind=np.hstack((ind_flex,ind_ext))
#############

ranges={}
means={}
label_list=['rest','Little\n finger:\n flex','Little\nfinger:\n extend','Ring\nfinger:\n flex','Ring\nfinger:\n extend','Middle\nfinger:\n flex','Middle\nfinger:\n extend','Index\nfinger:\n flex','Index\nfinger:\n extend','Thumb:\n down','Thumb:\n up','Thumb:\n left','Thumb:\n right'
            ,'Wrist: bend','Wrist: rotate anti-clockwise','Wrist: rotate clockwise','Little finger: bend+Ring finger: bend','	Little finger: bend+Thumb: down','Little finger: bend+Thumb: left','Little finger: bend+thumb: right','	Little finger: bend+wrist: bend'
            ,'Little finger: bend+Wrist: stretch','Little finger: bend+Wrist: rotate anti-clockwise','	Little finger: bend+Wrist: rotate clockwise','Ring finger: bend+Middle finger: bend','	Ring finger: bend+Thumb: down','Ring finger: bend+Thumb: left','	Ring finger: bend+Thumb: right'
            ,'	Ring finger: bend+Wrist: bend','Ring finger: bend+Wrist: stretch','Ring finger: bend+Wrist: rotate anti-clockwise','Ring finger: bend+Wrist: rotate clockwise','	Middle finger: bend+Index finger: bend','	Middle finger: bend+Thumb: down','	Middle finger: bend+Thumb: left'
            ,'	Middle finger: bend+Thumb: right','Middle finger: bend+Wrist: bend','Middle finger: bend+Wrist: stretch','Middle finger: bend+Wrist: rotate anti-clockwise','Middle finger: bend+Wrist: rotate clockwise','	Index finger: bend+Thumb: down','Index finger: bend+Thumb: left'
            ,'Index finger: bend+Thumb: right','Index finger: bend+Wrist: bend','	Index finger: bend+Wrist: stretch','Index finger: bend+Wrist: rotate anti-clockwise','	Index finger: bend+Wrist: rotate clockwise','	Thumb: down+Thumb: left','	Thumb: down+Thumb: right','Thumb: down+Thumb:bend'
            ,'Thumb: down+Thumb:stretch','Thumb: down+Wrist: rotate anti-clockwise','	Thumb: down+Wrist: rotate clockwise','	Wrist: bend+Wrist: rotate anti-clockwise','	Wrist: bend+Wrist: rotate clockwise','	Wrist: stretch+Wrist: rotate anti-clockwise','Wrist: stretch+Wrist: rotate clockwise'
            ,'Extend all fingers (without thumb)','	All fingers: bend (without thumb)','	Extend all fingers (without thumb)','	Palmar grasp' ,'	Wrist: rotate anti-clockwise with the Palmar grasp','	Pointing (index: stretch, all other: bend)','	3-digit pinch','3-digit pinch with Wrist: anti-clockwise rotation','Key grasp with Wrist: anti-clockwise rotation','']


force_labels=['index', 'middle','ring','little','thumb left-right','thumb up-down','thumb accumulated']
finger_colors = ['black','blue','red','grey','teal','sienna','teal']
color_list=['red','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow','black','yellow']


#############
finger_1=1
finger_2=2
fig2, ax2 = plt.subplots(2,1)
for k in range(finger_1,finger_2):
    ax2[0].plot(normalized_stiffness[:,k],label=force_labels[k],color=finger_colors[k])
ax2[0].set_ylabel('Estimated normalized stiffness (%)',fontsize=15)
ax2[0].set_yticks(np.arange(0, 110, 10),['0','10','20','30','40','50','60','70','80','90','100'])

ax2[0].set_ylim(0, 110)
#ax2[0].set_xlabel('time (epoch)')
#plt.plot(peaks, normalized_stiffness[:,peaks], "x")

for k in range(finger_1,finger_2):
    ax2[1].plot(normalized_force[:,k],label=force_labels[k],color=finger_colors[k])
ax2[1].set_xlabel('time (epoch)',fontsize=15)
ax2[1].set_ylabel('Force percentage (%)',fontsize=15)
for i in np.unique(movement_id):
    ranges[i] = np.where(movement_id == int(i))
    data_range=np.r_[ranges[i][0]]
    means[i]= np.round(np.mean(normalized_stiffness[data_range],axis=0),2)
    ax2[0].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1, color=color_list[np.where(np.unique(movement_id)==i)[0][0]],label=label_list[int(i)])
    ax2[1].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1, color=color_list[np.where(np.unique(movement_id) == i)[0][0]])
    ax2[0].annotate(label_list[int(i) ], xy=(int(ranges[i][0][0]), 100), fontsize=15)
    #ax2[1].annotate(label_list[int(i) ], xy=(int(ranges[i][0][0]), 100), fontsize=15)
df = pd.DataFrame.from_dict(means).T
df.to_excel('means1.xlsx')
### Means
# for i in means:
#     for k in range(finger_1,finger_2):
#         mean_value=means[i][k]
#         ax2[0].plot((ranges[i][0][0], ranges[i][0][-1]), (means[i][k], means[i][k]), finger_colors[k],'-.', linewidth=2)
        #ax2[0].axhline(y = means[i][k], xmin=ranges[i][0][0],xmax=ranges[i][0][-1], color = 'r', linestyle = '-')

#box = ax2[1].get_position()
#ax2[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax2[1].legend(loc='lower left',fontsize=15) #, bbox_to_anchor=(1, 0.5)
ax2[1].set_ylim(-100, 110)

##########

rms_features=np.hstack((rms_flex,rms_ext))
wl_features=np.hstack((wl_flex,wl_ext))
tdf4=np.hstack((rms_features,wl_features))
tp_features=np.hstack((tp_flex,tp_ext))
sm_features=np.hstack((sm_flex,sm_ext))
fd2= np.hstack((tp_features,sm_features))
tfdf=np.hstack((fd2,tdf4))

#classifier(tfdf,movement_id,5)
from sklearn.linear_model import LinearRegression, Ridge,ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
conc2=np.concatenate((rms_flex, rms_ext),axis=1)

X=tfdf#rms_features#force[:,0:6]#rms_features##tfdf#tdf4##conc2#
if target=='f':
    y=force[:,0:6]
else:
    y=normalized_stiffness[:,0:6]#
#force=normalized_stiffness
#X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
reg={}
predicted_force={}
score=pd.DataFrame()#{}
coeff=[]
if to_train==1:
    for i in range(6):
        print(f'Regression for {i+1}')
        reg[i] = MLPRegressor(activation='tanh',random_state=1, max_iter=500).fit(X, y[:,i])## LinearRegression().fit(X, y[:,i])
        #predicted_force[i]=butter_lowpass_filter(reg[i].predict(X),2048/1000,2048/50)#reg[i].predict(X)#
        #score[i]=round(r_square(y[:,i], predicted_force[i]),2)
        #coeff.append(reg[i].coef_)
    #save the model to disk
    pickle.dump(reg, open(filename, 'wb'))
if to_train==0:
    # # load the model from disk
    reg = pickle.load(open(filename, 'rb'))


for i in range(6):

    predicted_force[i]=reg[i].predict(X)#butter_lowpass_filter(,2048/2000,2048/50)#butter_lowpass_filter(reg[i].predict(X_p.reshape(1, -1)),2048/1000,2048/50)#
    #score[i]=[round(vaf(y[:,i], predicted_force[i]),2),round(r_square(y[:,i], predicted_force[i]),2) ,round(rmspe(y[:,i], predicted_force[i]),2) ,round(rmse(y[:,i], predicted_force[i]),2),round(mape(y[:,i], predicted_force[i]),2) ,round(nrmse(y[:,i], predicted_force[i]),4) ]
    #coeff.append(reg[i].coef_)
moav_predictied=np.empty((predicted_force[1].shape[0],6))
moav_y=np.empty((predicted_force[1].shape[0],6))
index=0
for i in range(6):
    moav_predictied[:,index]=moving_average(predicted_force[i][:],20)
    index=index+1
predicted_force=moav_predictied.T
index=0
for i in y.T:
    moav_y[:,index]=moving_average(i,20)
    index=index+1
force=moav_y
y=moav_y
for i in range(6):
    score = score.append(evaluate_regression_metrics(predicted_force[i],y[:, i],force_labels[i]))
       #[round(vaf(y[:, i], predicted_force[i]), 2), round(r_square(y[:, i], predicted_force[i]), 2),
       #         round(rmspe(y[:, i], predicted_force[i]), 2), round(rmse(y[:, i], predicted_force[i]), 2),
       #         round(mape(y[:, i], predicted_force[i]), 2), round(nrmse(y[:, i], predicted_force[i]), 4)]

score.to_excel('regression_'+target+'1.xlsx')
print(score)
fig2, ax2 = plt.subplots(6)
ax2[0].plot(predicted_force[0],'--',label='index estimated')
ax2[0].plot(force[:,0],label='index experimental')
ax2[1].plot(predicted_force[1],'--',label='middle estimated')
ax2[1].plot(force[:,1],label='middle experimental')
ax2[2].plot(predicted_force[2],'--',label='ring estimated')
ax2[2].plot(force[:,2],label='ring experimental')
ax2[3].plot(predicted_force[3],'--',label='little estimated')
ax2[3].plot(force[:,3],label='little experimental')
ax2[4].plot(predicted_force[4],'--',label='thumb left/right estimated')
ax2[4].plot(force[:,4],label='thumb left/right experimental')
ax2[-1].set_xlabel('time')
ax2[5].plot(predicted_force[5],'--',label='thumb up/down estimated')
ax2[5].plot(force[:,5],label='thumb up/down experimental')
#fig2.suptitle(f'Extracted stiffness from force vs Estimated stiffness from EMG ') #R2={score}
#fig2.suptitle(f'Measured force vs Estimated force from EMG ') #R2={score}

for ax in ax2:
    ax.legend(loc=7)
for i in np.unique(movement_id):
    ranges[i] = np.where(movement_id == int(i))
    #data_range=np.r_(ranges[i])

    ax2[0].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1, color=color_list[np.where(np.unique(movement_id)==i)[0][0]],label=label_list[int(i)])
    ax2[0].annotate(label_list[int(i) ], xy=(int(ranges[i][0][0]), 6), fontsize=10)
    ax2[1].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1, color=color_list[np.where(np.unique(movement_id) == i)[0][0]])
    ax2[2].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1,
                   color=color_list[np.where(np.unique(movement_id) == i)[0][0]])
    ax2[3].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1,
                   color=color_list[np.where(np.unique(movement_id) == i)[0][0]])
    ax2[4].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1,
                   color=color_list[np.where(np.unique(movement_id) == i)[0][0]])
    ax2[5].axvspan(int(ranges[i][0][0]), int(ranges[i][0][-1]), alpha=0.1,
                   color=color_list[np.where(np.unique(movement_id) == i)[0][0]])


if target=='f':
    fig2.text(0.09, 0.5, 'Force (N)', va='center', rotation='vertical',fontsize=14)
else:
    fig2.text(0.09, 0.5, 'Stiffness (%)', va='center', rotation='vertical',fontsize=14)




figname='fig/est_'+target+'_'+'1.pdf'
fig2.savefig(figname, format='pdf')
plt.show()

# ## Confidence plot
# confidence=0.5
# for i in range(6):
#     ci = confidence * np.std(force[:,i]) / np.mean(force[:,i])
#     ax2[i].fill_between(np.arange(0,force.shape[0],1),force[:,i]-ci,force[:,i]+ci, alpha = 0.1, color = 'blue')


#
# with open('data_s1_58_61.pkl', 'rb') as handle:
#     data2 = pickle.load(handle)
#
#
# rms_flex, mav_flex, iav_flex, var_flex, wl_flex, mf_flex, pf_flex, mp_flex, tp_flex, sm_flex, msf_flex,rms_ext, mav_ext, iav_ext, var_ext, wl_ext, mf_ext, pf_ext, mp_ext, tp_ext, sm_ext, msf_ext, movement_id, force, stiffness_estimation, estimation_r2 = list(map(data2.get, ['rms_flex', 'mav_flex', 'iav_flex', 'var_flex', 'wl_flex', 'mf_flex', 'pf_flex', 'mp_flex', 'tp_flex', 'sm_flex', 'msf_flex','rms_ext', 'mav_ext', 'iav_ext', 'var_ext', 'wl_ext', 'mf_ext', 'pf_ext', 'mp_ext', 'tp_ext', 'sm_ext', 'msf_ext', 'movement_id','force','stiffness','r2']) )
# normalized_stiffness=stiffness_estimation#normalize(stiffness_estimation, axis=0, norm='max')#
# moav_stiffness=np.empty((normalized_stiffness.shape[0],normalized_stiffness.shape)[1])
# index=0
# for i in stiffness_estimation.T:
#     moav_stiffness[:,index]=moving_average(i,100)
#     index=index+1
# #scaler = MinMaxScaler(feature_range=(0, 100))
# #scaler.fit(moav_stiffness)
# #normalized_stiffness=scaler.transform(moav_stiffness)#normalize(moav_stiffness, axis=0, norm='l1')#normalized_stiffness#moav_stiffness#
# shape1, shape2 = moav_stiffness.shape
# moav_stiffness2 = moav_stiffness.reshape(-1, 1)
# scaler = MinMaxScaler(feature_range=(0, 100))
# scaler.fit(moav_stiffness2)
# normalized_stiffness= scaler.transform(moav_stiffness2).reshape(shape1, shape2)
# rms_features=np.hstack((rms_flex,rms_ext))
# wl_features=np.hstack((wl_flex,wl_ext))
# tdf4=np.hstack((rms_features,wl_features))
# tp_features=np.hstack((tp_flex,tp_ext))
# sm_features=np.hstack((sm_flex,sm_ext))
# fd2= np.hstack((tp_features,sm_features))
# tfdf=np.hstack((fd2,tdf4))
# #normalized_stiffness=normalize(stiffness_estimation, axis=0, norm='max')
# X=tfdf#rms_features#tfdf#tfdf##tdf4##conc2#
#
# y=normalized_stiffness[:,0:6]#force[:,0:6]#
# #force=normalized_stiffness
# print(y.shape)
# for i in range(6):
#     predicted_force[i]=butter_lowpass_filter(reg[i].predict(X),2048/1000,2048/50)#,2048/500,2048/50)##np.matmul(X,coeff[i].reshape(-1,1))#butter_lowpass_filter(,2048/1000,2048/50)#
#     score[i]=round(r_square(force[:,i],predicted_force[i]),2)#reg[i].score(X*coeff[i], y[:,i])
# predicted_force=np.array([predicted_force[item] for item in predicted_force])
# # scaler = MinMaxScaler(feature_range=(-100, 100))
# # scaler.fit(predicted_force)
# # predicted_force=scaler.transform(predicted_force)
# # scaler = MinMaxScaler(feature_range=(-100, 100))
# # scaler.fit(force)
# # force=scaler.transform(force)
# #predicted_force2=resample(predicted_force,n_samples=predicted_force.shape[0]*50)
# stiffness_estimation=force
# estimated_stiffness2=predicted_force#force_stiffness(predicted_force2,50)
# print(stiffness_estimation[0].shape)
# fig3, ax3 = plt.subplots(6)
# ax3[0].plot(estimated_stiffness2[0],'--',label='index predicted')
# ax3[0].plot(stiffness_estimation[:,0],label='index')
# ax3[1].plot(estimated_stiffness2[1],'--',label='middle predicted')
# ax3[1].plot(stiffness_estimation[:,1],label='middle')
# ax3[2].plot(estimated_stiffness2[2],'--',label='ring predicted')
# ax3[2].plot(stiffness_estimation[:,2],label='ring')
# ax3[3].plot(estimated_stiffness2[3],'--',label='pinky predicted')
# ax3[3].plot(stiffness_estimation[:,3],label='pinky')
# ax3[4].plot(estimated_stiffness2[4],'--',label='thumb f/e predicted')
# ax3[4].plot(stiffness_estimation[:,4],label='thumb f/e')
# ax3[-1].set_xlabel('time')
# ax3[5].plot(estimated_stiffness2[5],'--',label='thumb a/a predicted')
# ax3[5].plot(stiffness_estimation[:,5],label='thumb a/a')
# fig3.suptitle(f'Measured vs estimated stiffness, R2={score}')
# for ax in ax3:
#     ax.legend(loc='upper left')
# scaler = MinMaxScaler(feature_range=(-100, 100))
# scaler.fit(force)
#print()

