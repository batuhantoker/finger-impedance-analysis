# Functions used to analyze EMG/force data

from sklearn.preprocessing import MinMaxScaler
import pickle
from itertools import chain
import math
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfilt, sosfreqz, lfilter, freqz
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from numpy.fft import fft, ifft
from scipy.optimize import curve_fit
from finger_impedance.core.tfestimate import *
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import plot_confusion_matrix

np.seterr(divide="ignore")


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def zero_lag_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def data_preprocess(emg_data, fs, lowcut, highcut, cutoff):
    print("Data filtering...")
    emg_ = zero_lag_filter(emg_data, lowcut, highcut, fs, order=4)
    emg_ = abs(emg_)
    # emg_ = butter_lowpass_filter(emg_rect, cutoff, fs, 4)
    # emg_=samplerate.resample(emg_,0.5)
    return emg_


def rolling_rms(x):
    N = 150
    xc = np.cumsum(abs(x) ** 2)
    return np.sqrt((xc[N:] - xc[-N]) / N)


def mape(real, estimate):
    return mean_absolute_percentage_error(real, estimate)


def rmse(real, estimate):
    return mean_squared_error(real, estimate, squared=False)


def nrmse1(real, estimate):
    return rmse(real, estimate) / (np.max(np.abs(real)) - np.min(np.abs(real)))


def nrmse2(real, estimate):
    return rmse(real, estimate) / (np.max(np.abs(real)))


def rmspe(real, estimate):
    return np.linalg.norm(estimate - real) / np.sqrt(len(real))


def vaf(real, estimate):
    return 100 * (
        1 - (np.var(real - estimate) / np.var(estimate))
    )  # 100*(1-(np.sum((real-estimate)**2)/np.sum((estimate**2))))


def r_square(real, estimate):
    residuals = real - estimate
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((real - np.mean(real)) ** 2)
    return 1 - (ss_res / ss_tot)


def force_mean(data, epoch):
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(
        data[0 : number_of_segments * epoch, :], number_of_segments
    )
    RMS = np.empty([number_of_segments, data.shape[1]])
    for i in range(number_of_segments):
        RMS[i, :] = np.mean(
            splitted_data[i], axis=0
        )  # np.sqrt(np.mean(np.square(splitted_data[i]), axis=0))
    return RMS


def force_window(splitted_data):
    new_data = []

    for i in range(len(splitted_data)):
        if i == 0:
            new_data.append(
                np.concatenate(
                    (splitted_data[i], splitted_data[i + 1], splitted_data[i + 2])
                )
            )
        if i != 0 and i != len(splitted_data) - 2 and i != len(splitted_data) - 1:
            new_data.append(
                np.concatenate(
                    (splitted_data[i - 1], splitted_data[i], splitted_data[i + 1])
                )
            )
        if (
            i == len(splitted_data)
            or i == len(splitted_data) - 1
            or i == len(splitted_data) - 2
        ):
            new_data.append(
                np.concatenate(
                    (splitted_data[i], splitted_data[i - 1], splitted_data[i - 2])
                )
            )
        # print(i,len(splitted_data))
    return new_data


def stiffness(freq, K):  #
    return np.absolute(-K * 1j / (freq))


def bode_plot(w, mag):
    plt.title("Bode magnitude plot")
    plt.semilogx(w, mag, "x")
    plt.grid()


def feature_extraction(data, epoch):
    print("Feature extraction...")
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
    # max_ind = np.empty([number_of_segments, 4])
    for i in range(number_of_segments):
        RMS[i, :] = np.sqrt(np.mean(np.square(splitted_data[i]), axis=0))
        # max_ind [i,:] = RMS[i,:][np.argpartition(RMS[i,:],5, axis=0)]
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
    return RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM  # , max_ind


def class_map(data, epoch):
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0 : number_of_segments * epoch], number_of_segments)
    class_value = np.empty([number_of_segments])
    for i in range(number_of_segments):
        class_value[i] = np.mean(splitted_data[i])
    return class_value


def classification_report_with_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)  # return accuracy score


def classifier(features, labels, k_fold):
    Y = labels
    X = features
    number_of_k_fold = k_fold
    random_seed = 42
    outcome = []
    model_names = []
    # Variables for average classification report
    originalclass = []
    classification = []
    models = [
        ("LogReg", LogisticRegression()),
        ("SVM", SVC()),
        # ('DecTree', DecisionTreeClassifier()),
        # ('KNN', KNeighborsClassifier(n_neighbors=15)),
        # ('LinDisc', LinearDiscriminantAnalysis()),
        # ('GaussianNB', GaussianNB()),
        # ('MLPC', MLPClassifier(activation='relu', solver='adam', max_iter=500)),
        # ('RFC',RandomForestClassifier()),
        # ('ABC', AdaBoostClassifier())
    ]

    for model_name, model in models:
        k_fold_validation = model_selection.KFold(
            n_splits=number_of_k_fold, random_state=random_seed, shuffle=True
        )
        results = model_selection.cross_val_score(
            model,
            X,
            Y,
            cv=k_fold_validation,
            scoring=make_scorer(classification_report_with_accuracy_score),
        )
        outcome.append(results)
        model_names.append(model_name)
        output_message = "%s| Mean=%f STD=%f" % (
            model_name,
            results.mean(),
            results.std(),
        )
        print(output_message)
    print(classification)
    fig = plt.figure()
    fig.suptitle("Machine Learning Model Comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(outcome)
    plt.ylabel("Accuracy [%]")
    ax.set_xticklabels(model_names)
    fig2 = plt.figure()
    plt.show()
    # plt.savefig('myimage.png', format='png')


def force_stiffness(data, epoch):
    print("Stiffness estimation...")
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(
        data[0 : number_of_segments * epoch, :], number_of_segments
    )
    # splitted_data = force_window(splitted_data)
    RMS = np.empty([number_of_segments, data.shape[1]])
    stiffness_estimation = np.empty([number_of_segments, data.shape[1]])
    estimation_r2 = np.empty([number_of_segments, data.shape[1]])
    sr = 2048
    # sampling interval
    ts = 1.0 / sr

    for i in range(number_of_segments):
        k = []
        r2 = []
        # RMS[i, :] = np.mean(splitted_data[i], axis=0)#np.sqrt(np.mean(np.square(splitted_data[i]), axis=0))
        for j in range(RMS.shape[1]):
            x = splitted_data[i][:, j]
            y = np.ones(len(x))  # *-1
            tf = tfest(y, x)
            tf.estimate(0, 0, sr, method="fft")
            w1, mag1 = tf.bode_estimate()

            k.append(mag1[0])

        stiffness_estimation[i, :] = k
    return stiffness_estimation


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret = np.append(ret, [ret[-1] * (np.ones(n - 1))])
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def pct_change(df):
    pct = 1 - df.iloc[0] / df
    return pct


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def force_stiffness_v0(data, epoch):
    print("Stiffness estimation...")
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(
        data[0 : number_of_segments * epoch, :], number_of_segments
    )
    splitted_data = force_window(splitted_data)
    RMS = np.empty([number_of_segments, data.shape[1]])
    stiffness_estimation = np.empty([number_of_segments, data.shape[1]])
    estimation_r2 = np.empty([number_of_segments, data.shape[1]])
    sr = 2048
    # sampling interval
    ts = 1.0 / sr

    for i in range(number_of_segments):
        k = []
        r2 = []
        # RMS[i, :] = np.mean(splitted_data[i], axis=0)#np.sqrt(np.mean(np.square(splitted_data[i]), axis=0))
        for j in range(RMS.shape[1]):
            x = splitted_data[i][:, j]
            X = fft(x)
            X = np.nan_to_num(X)  # Converts NaN to 0
            N = len(X)
            n = np.arange(N)
            T = N / sr
            freq = np.fft.fftfreq(len(x), ts) * 6.28
            t = np.arange(0, ts * N, ts)
            # plt.figure(figsize = (12, 6))
            X = X[1:]
            freq = freq[1:]
            X = X[: len(X) // 2]
            freq = freq[: len(freq) // 2]

            # plt.subplot(121)
            # amp = 20 * np.log10((np.absolute(X)))

            popt, pcov = curve_fit(stiffness, freq, np.absolute(X))
            rsquare = r_square(np.absolute(X), stiffness(freq, *popt))
            k.append(popt[0])
            r2.append(rsquare)
        stiffness_estimation[i, :] = k
        estimation_r2[i, :] = r2
    return stiffness_estimation, estimation_r2


def evaluate_regression_metrics(y_pred, y_true, index):
    # Calculate mean/median of prediction
    mean_pred = np.mean(y_pred)
    median_pred = np.median(y_pred)

    # Calculate standard deviation of prediction
    std_pred = np.std(y_true - y_pred)

    # Calculate range of prediction
    range_pred = max(y_pred) - min(y_pred)

    # Calculate coefficient of determination (R2)
    r2 = r2_score(y_true, y_pred)

    # Calculate relative standard deviation/coefficient of variation (RSD)
    rsd = std_pred / mean_pred

    # Calculate relative squared error (RSE)
    rse = np.sqrt(mean_squared_error(y_true, y_pred)) / mean_pred

    # Calculate mean absolute error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate relative absolute error (RAE)
    rae = mae / mean_pred

    # Calculate mean squared error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Calculate root mean squared error on prediction (RMSE/RMSEP)
    rmsep = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate normalized root mean squared error (norm RMSEP)
    norm_rmsep = rmsep / mean_pred

    # vaf
    vaf_value = vaf(y_true, y_pred)

    # Calculate relative root mean squared error (RRMSEP)
    rrmsep = rmsep / std_pred

    # Calculate relative root mean squared error (RRMSEP)
    nrmse_value1 = nrmse1(y_true, y_pred)
    nrmse_value2 = nrmse2(y_true, y_pred)

    # Create a dictionary of the regression metrics
    metrics = {
        "R2": r2 * 100,
        "MAE": mae,
        "MSE": mse,
        "RMSEP": rmsep,
        "vaf": vaf_value,
        "RMSE": rmsep,
        "nRMSE1": nrmse_value1 * 100,
        "nRMSE2": nrmse_value2 * 100,
    }

    # Create a Pandas DataFrame of the regression metrics
    metrics_df = pd.DataFrame(metrics, index=[index])

    # Return the DataFrame
    return metrics_df
