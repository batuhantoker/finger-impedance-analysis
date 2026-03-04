import pickle
import numpy as np
from scipy import signal
import math
from sklearn.cluster import MeanShift, estimate_bandwidth
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def activation_map(data, epoch):
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(
        data[0 : number_of_segments * epoch, :], number_of_segments
    )
    RMS = np.empty([number_of_segments, data.shape[1]])
    MAV = np.empty([number_of_segments, data.shape[1]])
    IAV = np.empty([number_of_segments, data.shape[1]])
    VAR = np.empty([number_of_segments, data.shape[1]])
    WL = np.empty([number_of_segments, data.shape[1]])
    MF = np.empty([number_of_segments, data.shape[1]])
    PF = np.empty([number_of_segments, data.shape[1]])
    MP = np.empty([number_of_segments, data.shape[1]])
    TP = np.empty([number_of_segments, data.shape[1]])
    SM = np.empty([number_of_segments, data.shape[1]])
    for i in range(number_of_segments):
        RMS[i, :] = np.sqrt(np.mean(np.square(splitted_data[i]), axis=0))
        MAV[i, :] = np.mean(np.abs(splitted_data[i]), axis=0)
        IAV[i, :] = np.sum(np.abs(splitted_data[i]), axis=0)
        VAR[i, :] = np.var(splitted_data[i], axis=0)
        WL[i, :] = np.sum(np.diff(splitted_data[i], prepend=0), axis=0)
        freq, power = signal.periodogram(splitted_data[i], axis=0)
        fp = np.empty([len(freq), power.shape[1]])
        for k in range(len(freq)):
            fp[k] = power[k, :] * freq[k]
        MF[i, :] = np.sum(fp, axis=0) / np.sum(power, axis=0)  # Mean frequency
        PF[i, :] = freq[np.argmax(power, axis=0)]  # Peak frequency
        MP[i, :] = np.mean(power, axis=0)  # Mean power
        TP[i, :] = np.sum(power, axis=0)  # Total power
        SM[i, :] = np.sum(fp, axis=0)  # Spectral moment
    return RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM


def class_map(data, epoch):
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0 : number_of_segments * epoch], number_of_segments)
    class_value = np.empty([number_of_segments])
    for i in range(number_of_segments):
        class_value[i] = np.sqrt(np.mean(np.square(splitted_data[i])))
    return class_value


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def mean_shift_feature(data_emg):
    # data_emg = np.reshape(data_emg, (len(data_emg), 8, 8))
    # data_emg = data_emg.astype(np.float64)
    labels = np.zeros((len(data_emg), 8, 8))
    for i in range(len(data_emg)):
        data = data_emg[i, :]
        data = np.reshape(data, (8, 8))
        # resized_img = np.uint8((255 * (resized_img - np.min(resized_img)) / np.ptp(resized_img)).astype(int))
        flat_image = np.reshape(data, [-1, 1])
        # Estimate bandwidth
        bandwidth2 = estimate_bandwidth(flat_image, quantile=0.1, n_samples=2500)
        ms = MeanShift(bandwidth=bandwidth2)
        ms.fit(flat_image)
        labels[i, :, :] = np.reshape(ms.labels_, [8, 8])  # np.asarray(ms.labels_)
    labels = np.asarray(np.reshape(labels, (len(data_emg), 64)), dtype=object)
    return labels


def data_reshape(data):
    data = np.reshape(data, (len(data), 64))  # 8, 8
    data = data.astype(np.float64)
    return data


# load data
flex_pp = data_reshape(np.loadtxt("flex_pp.txt"))
ext_pp = data_reshape(np.loadtxt("ext_pp.txt"))
emg_class = np.loadtxt("emg_class.txt")

epoch = 100
emg_class = class_map(emg_class, epoch)
valid_classes = np.r_[np.array([i for i, v in enumerate(emg_class) if v.is_integer()])]


RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM = activation_map(ext_pp, epoch)
# MSF = mean_shift_feature(RMS)
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
RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM = activation_map(flex_pp, epoch)
# MSF = mean_shift_feature(RMS)
dict_flex = {
    "rms_flex": RMS,
    "mav_flex": MAV,
    "iav_flex": IAV,
    "var_flex": VAR,
    "wl_flex": WL,
    "mf_flex": MF,
    "pf_flex": PF,
    "mp_flex": MP,
    "tp_flex": TP,
    "sm_flex": SM,
}  # , 'msf_flex':MSF
dict_target = {"movement_id": emg_class}


z = dict(dict_flex, **dict_ext)
z2 = dict(z, **dict_target)
print(z2.keys())
with open("data11.pkl", "wb") as handle:
    pickle.dump(z2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# data7 s1, data8 s2, data9 s3, data10 s4, data11 s1 50-63
