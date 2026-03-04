import mat73,h5py
from finger_impedance.core.functions import *
subject_range = chain(range(1,5), range(6, 21))
for i in subject_range:
    subject=i
    print('Now subject number is '+ f'{subject}')
    if i==5:
        data_dict = h5py.File('males/'+f's{subject}.mat')
    else:
        data_dict = mat73.loadmat('males/' + f's{subject}.mat')

    movement=1
    movement_f=13
    movements=np.array(range(movement,movement_f))

    movement_ranges=np.in1d(data_dict['class'], movements).nonzero()[0]
    movement_ranges_0all=np.where(data_dict['class']==0)

    movement_ranges_0=movement_ranges_0all[0][0:(round(movement_ranges.shape[0]/(movement_f-movement)+1))]

    data_range=np.r_[movement_ranges_0,movement_ranges]
    data_range_0=np.r_[movement_ranges_0all]
    emg_ext=data_dict['emg_extensors'][data_range,:,:]
    emg_flex=data_dict['emg_flexors'][data_range,:,:]
    force=data_dict['force'][data_range,:]
    force = force*40-100
    emg_class =data_dict['class'][data_range]

    plt.plot(data_dict['force'][data_range_0,:])
    #plt.show()





    # Flattening for 3D
    emg_ext_reshaped=emg_ext.reshape(len(emg_ext),64)
    emg_flex_reshaped=emg_flex.reshape(len(emg_flex),64)

    # Filtering
    fs = 2048
    lowcut = 15
    highcut = 350
    window_size = 137
    cutoff = 60

    emg_ext_pp = data_preprocess(emg_ext_reshaped,fs,lowcut,highcut,cutoff)
    emg_flex_pp = data_preprocess(emg_flex_reshaped,fs,lowcut,highcut,cutoff)

    # Feature extraction
    epoch=50

    mean_force=force_mean(force,epoch)
    emg_class = class_map(emg_class,epoch)

    valid_classes=np.r_[np.array([i for i, v in enumerate(emg_class) if v.is_integer()])]

    RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM=feature_extraction(emg_ext_pp,epoch)
    dict_ext = {'rms_ext':RMS[valid_classes,:], 'mav_ext':MAV[valid_classes,:], 'iav_ext':IAV[valid_classes,:], 'var_ext':VAR[valid_classes,:],
                'wl_ext':WL[valid_classes,:], 'mf_ext':MF[valid_classes,:], 'pf_ext':PF[valid_classes,:], 'mp_ext':MP[valid_classes,:],
                'tp_ext':TP[valid_classes,:],'sm_ext':SM[valid_classes,:]} #, 'msf_ext':MSF
    RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM =feature_extraction(emg_flex_pp,epoch)
    dict_flex= {'rms_flex':RMS[valid_classes,:], 'mav_flex':MAV[valid_classes,:], 'iav_flex':IAV[valid_classes,:], 'var_flex':VAR[valid_classes,:],
                'wl_flex':WL[valid_classes,:], 'mf_flex':MF[valid_classes,:], 'pf_flex':PF[valid_classes,:], 'mp_flex':MP[valid_classes,:],
                'tp_flex':TP[valid_classes,:], 'sm_flex':SM[valid_classes,:]} # , 'msf_flex':MSF
    estimated_stiffness=force_stiffness(force,epoch)
    dict_target = {'movement_id': emg_class[valid_classes]}
    dict_force = {'force': mean_force[valid_classes,:]}
    dict_stiffness= {'stiffness': estimated_stiffness[valid_classes,:]}
    dict0 = dict(dict_flex, **dict_ext)
    dict1 = dict(dict0, **dict_target)
    dict2 = dict(dict1, **dict_stiffness)
    dict_final = dict(dict2, **dict_force)
    print(dict_final.keys())
    with open(f'males/data_s{subject}.pkl', 'wb') as handle:
        pickle.dump(dict_final, handle, protocol=pickle.HIGHEST_PROTOCOL)



