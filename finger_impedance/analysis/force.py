from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
import pickle, math
from numpy.fft import fft, ifft
from scipy.optimize import curve_fit
from matplotlib.mlab import psd, csd
from finger_impedance.core.functions import *
from finger_impedance.core.tfestimate import *
from numpy.lib.stride_tricks import sliding_window_view

np.seterr(divide="ignore")
from sklearn.preprocessing import MinMaxScaler

plt.rcParams["figure.figsize"] = (20, 20)
plt.rcParams.update({"font.size": 18})
plt.style.use("bmh")
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["figure.figsize"] = (20, 10)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def stiffness(freq, K):  #
    return np.absolute(-K * 1j / (freq))


def tfe(x, y, *args, **kwargs):
    """estimate transfer function from x to y, see csd for calling convention"""
    return csd(y, x, *args, **kwargs) / psd(x, *args, **kwargs)


def rms(data, epoch):
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


def r_square(real, estimate):
    residuals = real - estimate
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((real - np.mean(real)) ** 2)
    return 1 - (ss_res / ss_tot)


force = np.loadtxt("force.txt")
epoch = 100
window_size = epoch * 2
number_of_segments = math.trunc(len(force) / epoch)

# sampling rate
sr = 2048
# sampling interval
ts = 1.0 / sr
x = force[2000:2100, 2]
y = np.ones(len(x))  # *50#*0.1#*-1
plt.figure()
plt.plot(x)
plt.title("Finger force over an epoch")
# plt.legend(['Force'])
plt.xlabel("Time [sample]")
plt.ylabel("Force [N]")
# plt.figure()
# plt.plot(y)
# plt.legend(['Position'])
# plt.xlabel('Time [sample]')
# plt.ylabel('Position [Gesture]')
tf = tfest(y, x)
tf.estimate(0, 0, sr, method="fft")
w1, mag1 = tf.bode_estimate()
plt.figure()

print(mag1[0])
tf2 = tfest(y, x)
tf2.estimate(1, 0, sr, method="fft")
w2, mag2 = tf2.bode_estimate()
tf3 = tfest(y, x)
tf3.estimate(2, 0, sr, method="fft")
w3, mag3 = tf3.bode_estimate()
bode_plot(w2, mag1)
bode_plot(w2, mag2)
bode_plot(w2, mag3)

plt.legend(
    ["Spring estimation", "Spring-damper estimation", "Mass-spring-damper estimation"]
)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.show()

tf.plot_bode()
tf.estimate(2, 0, method="fft")
tf.plot_bode()

print(tf)
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
amp = 20 * np.log10((np.absolute(X)))  # 20*np.log10
popt, pcov = curve_fit(stiffness, freq, (np.absolute(X)))  # 20*np.log10
rsquare = r_square(np.absolute(X), stiffness(freq, *popt))

print(popt, rsquare)
fig, ax = plt.subplots()
ax.plot(freq, amp, "*k", label="Experimental data")
ax.plot(
    freq,
    20 * np.log10(stiffness(freq, *popt)),
    "r-",
    label=f"Curve fitted, R2={round(rsquare,2)}",
)  # 20*np.log10
ax.set_xlabel("Freq (rad/s)")
ax.set_ylabel("FFT Magnitude |F(freq)| [dB]")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.xlim(0, 60)
plt.show()

#
