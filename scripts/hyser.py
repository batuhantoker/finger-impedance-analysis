import mat73, math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import scipy
from scipy import signal
from scipy.signal import butter, sosfilt, sosfreqz, lfilter, freqz
import samplerate
import pickle

# wget -r -N -c -np https://physionet.org/files/hd-semg/1.0.0/mvc_dataset/subject03_session1/
from finger_impedance.core.functions import *

dataset = "1dof"
for i in range(9, 15):
    subject = "%02d" % i
    print("Now subject number is " + subject)
    ## Import data
    filename = "hyser/s" + subject + "_hyser_" + dataset
    print(filename)
    data_dict = scipy.io.loadmat(filename + ".mat")

    emg = data_dict["EMG_data"]

    force = data_dict["force_data"]

    emg_class = data_dict["label_data"]

    plt.plot(force)

    # Filtering
    fs = 2048
    lowcut = 15
    highcut = 350
    window_size = 137
    cutoff = 60

    emg = data_preprocess(emg, fs, lowcut, highcut, cutoff)

    # Feature extraction
    epoch = 50

    mean_force = force_mean(force, epoch)
    emg_class = class_map(emg_class, epoch)

    RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM = feature_extraction(emg, epoch)
    dict_ext = {
        "rms_ext": RMS,
        "mav_ext": MAV,
        "iav_ext": IAV,
        "var_ext": VAR,
        "wl_ext": WL,
        "mf_ext": MF,
        "pf_ext": PF,
        "mp_ext": MP,
        "tp_ext": TP,
        "sm_ext": SM,
    }  # , 'msf_ext':MSF
    estimated_stiffness = force_stiffness(force, epoch)
    dict_target = {"movement_id": emg_class}
    dict_force = {"force": mean_force}
    dict_stiffness = {"stiffness": estimated_stiffness}

    dict1 = dict(dict_ext, **dict_target)
    dict2 = dict(dict1, **dict_stiffness)
    dict_final = dict(dict2, **dict_force)
    print(dict_final.keys())
    with open(filename + ".pkl", "wb") as handle:
        pickle.dump(dict_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
