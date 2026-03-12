"""Force-based stiffness estimation pipeline.

Demonstrates stiffness extraction from force data using FFT-based transfer
function estimation and curve fitting to a spring model (K/s).
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import csd, psd
from numpy.fft import fft
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit

from finger_impedance.core.functions import bode_plot, stiffness
from finger_impedance.core.tfestimate import tfest

np.seterr(divide="ignore")

if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.rcParams.update({"font.size": 18})
    plt.style.use("bmh")
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["figure.figsize"] = (20, 10)

    force = np.loadtxt("force.txt")
    epoch = 100
    window_size = epoch * 2
    number_of_segments = math.trunc(len(force) / epoch)

    # sampling rate
    sr = 2048
    # sampling interval
    ts = 1.0 / sr
    x = force[2000:2100, 2]
    y = np.ones(len(x))
    plt.figure()
    plt.plot(x)
    plt.title("Finger force over an epoch")
    plt.xlabel("Time [sample]")
    plt.ylabel("Force [N]")
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

    print(tf)
    X = fft(x)
    X = np.nan_to_num(X)
    N = len(X)
    n = np.arange(N)
    T = N / sr
    freq = np.fft.fftfreq(len(x), ts) * 6.28
    t = np.arange(0, ts * N, ts)
    X = X[1:]
    freq = freq[1:]
    X = X[: len(X) // 2]
    freq = freq[: len(freq) // 2]

    amp = 20 * np.log10((np.absolute(X)))
    popt, pcov = curve_fit(stiffness, freq, (np.absolute(X)))
    rsquare = 1 - np.sum((np.absolute(X) - stiffness(freq, *popt)) ** 2) / np.sum(
        (np.absolute(X) - np.mean(np.absolute(X))) ** 2
    )

    print(popt, rsquare)
    fig, ax = plt.subplots()
    ax.plot(freq, amp, "*k", label="Experimental data")
    ax.plot(
        freq,
        20 * np.log10(stiffness(freq, *popt)),
        "r-",
        label=f"Curve fitted, R2={round(rsquare,2)}",
    )
    ax.set_xlabel("Freq (rad/s)")
    ax.set_ylabel("FFT Magnitude |F(freq)| [dB]")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
