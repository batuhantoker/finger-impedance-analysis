"""Interactive matplotlib visualization for 8x8 HD-sEMG grid data.

Displays extensor/flexor activation maps with slider-based time navigation,
contraction percentages, local maxima detection, and image feature extraction
(HoG, Harris corners, Canny edges, MeanShift clustering).
"""

import math
from typing import List, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
from matplotlib.widgets import Slider
from scipy.signal import butter, lfilter
from skimage.feature import (
    corner_harris,
    corner_peaks,
    hog,
    peak_local_max,
)
from skimage.transform import resize
from sklearn.cluster import MeanShift, estimate_bandwidth

# Module-level constants
contraction_index = ["EMG1", "EMG2", "co-contraction"]
epoch = 200


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Apply automatic Canny edge detection using median-based thresholds.

    Args:
        image: Input grayscale image as uint8 array.
        sigma: Threshold sensitivity factor around median intensity.

    Returns:
        Binary edge image.
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    return edged


def image_features(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], np.ndarray]:
    """Extract image-based features from an EMG activation map.

    Computes Harris corners, HoG descriptors, Canny edges, MeanShift
    clustering labels, center of gravity, and local maxima coordinates.

    Args:
        data: 2D activation map array.

    Returns:
        Tuple of (harris_coords, hog_image, canny_edges, resized_image,
                  meanshift_labels, center_of_gravity, max_coordinates).
    """
    resized_img = resize(data, (32, 32))
    resized_img = np.uint8((255 * (resized_img - np.min(resized_img)) / np.ptp(resized_img)).astype(int))
    coords_harris = corner_peaks(corner_harris(resized_img))
    canny_edges = auto_canny(resized_img)
    fd, hog_image = hog(resized_img, visualize=True)
    flat_image = np.reshape(resized_img, [-1, 1])
    bandwidth2 = estimate_bandwidth(flat_image, quantile=0.1, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth2)
    ms.fit(flat_image)
    labels = np.reshape(ms.labels_, [32, 32])
    cog = scipy.ndimage.center_of_mass(resized_img)
    coordinates_max = peak_local_max(resized_img, min_distance=1, num_peaks=3)
    return coords_harris, hog_image, canny_edges, resized_img, labels, cog, coordinates_max


def data_reshape(data: np.ndarray) -> np.ndarray:
    """Reshape flat EMG data into (n_samples, 8, 8) grid format."""
    data = np.reshape(data, (len(data), 8, 8))
    data = data.astype(np.float64)
    return data


def mvc_calculator(ext: np.ndarray, flex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute maximum voluntary contraction for each channel.

    Args:
        ext: Extensor activation data.
        flex: Flexor activation data.

    Returns:
        Tuple of (mvc_extensor, mvc_flexor) arrays.
    """
    mvc_ext = np.amax(ext, axis=0)
    mvc_flex = np.amax(flex, axis=0)
    return mvc_ext, mvc_flex


def butter_lowpass(cutoff: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Design a lowpass Butterworth filter (local copy for standalone use)."""
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    """Apply a lowpass Butterworth filter (local copy for standalone use)."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def intensity_max(data: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute per-frame intensity and maximum intensity across all frames.

    Args:
        data: 3D array of shape (n_frames, rows, cols).

    Returns:
        Tuple of (intensity_per_frame, max_intensity).
    """
    It = np.empty(len(data))
    for i in range(len(data)):
        It[i] = np.sum(data[i, :, :])
    It_max = np.amax(It)
    return It, It_max


def mean_activation(am: np.ndarray) -> np.ndarray:
    """Compute mean activation map across all frames."""
    return np.mean(am, axis=0)


def local_maximum_pos(
    data: np.ndarray, threshold: float, neighborhood_size: int
) -> Tuple[List[float], List[float]]:
    """Find local maximum positions in a 2D array.

    Args:
        data: 2D input array.
        threshold: Minimum difference between local max and min to qualify.
        neighborhood_size: Size of the maximum/minimum filter window.

    Returns:
        Tuple of (x_positions, y_positions) of detected maxima.
    """
    data_max = scipy.ndimage.maximum_filter(data, neighborhood_size)
    maxima = data == data_max
    data_min = scipy.ndimage.minimum_filter(data, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0

    labeled, num_objects = scipy.ndimage.label(maxima)
    slices = scipy.ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)
    return x, y


def activation_map(data: np.ndarray, epoch: int) -> np.ndarray:
    """Compute RMS-based activation maps per epoch.

    Args:
        data: 3D array of shape (n_samples, rows, cols).
        epoch: Number of samples per epoch window.

    Returns:
        Activation maps of shape (n_segments, rows, cols).
    """
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0 : number_of_segments * epoch, :, :], number_of_segments)
    AM = np.empty([number_of_segments, data.shape[1], data.shape[2]])
    for i in range(number_of_segments):
        AM[i, :, :] = np.sqrt(np.mean(np.square(splitted_data[i]), axis=0))
    return AM


def class_map(data: np.ndarray, epoch: int) -> np.ndarray:
    """Compute epoch-wise RMS for class/label signals."""
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0 : number_of_segments * epoch], number_of_segments)
    class_value = np.empty([number_of_segments])
    for i in range(number_of_segments):
        class_value[i] = np.sqrt(np.mean(np.square(splitted_data[i])))
    return class_value


if __name__ == "__main__":
    # Load and preprocess data
    ext_raw = data_reshape(np.loadtxt("ext_raw.txt"))
    ext_pp = data_reshape(np.loadtxt("ext_pp.txt"))

    flex_raw = data_reshape(np.loadtxt("flex_raw.txt"))
    flex_pp = data_reshape(np.loadtxt("flex_pp.txt"))

    emg_class = np.loadtxt("emg_class.txt")

    def map_values(
        i: int, epoch: int
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray, float, str]:
        """Get activation map values and metadata for a given epoch index."""
        if emg_class[i * epoch] == 0:
            current_action = "rest"
        else:
            current_action = f"performing gesture {int(emg_class[i * epoch])}"
        Z0 = It_cc[i]
        Z = It_ext[i]
        Z1 = It_flex[i]
        Z2 = flex_pp[i, :, :]
        Z3 = ext_pp[i, :, :]
        time_sec = round(i * 0.00048828125 * epoch, 3)
        return Z0, Z, Z1, Z2, Z3, time_sec, current_action

    def update(val: float) -> None:
        """Update the interactive plot when the slider value changes."""
        current_i = s_factor.val
        global cbar_lim2, neighborhood_size, threshold_max
        Z0, Z, Z1, Z2, Z3, time_sec, current_action = map_values(
            round(current_i * len(ext_pp) / 100), epoch
        )
        typedraw = "none"
        im3 = axs[1].imshow(np.asarray(Z3), interpolation=typedraw)
        im3.set_clim(vmin=0, vmax=np.max(cbar_lim2))
        im4 = axs[2].imshow(np.asarray(Z2), interpolation=typedraw)
        im4.set_clim(vmin=0, vmax=np.max(cbar_lim2))
        im3.set_cmap("jet")
        im4.set_cmap("jet")
        x_ext, y_ext = local_maximum_pos(Z3, threshold_max * 0.2, neighborhood_size)
        x_flex, y_flex = local_maximum_pos(Z2, threshold_max * 0.2, neighborhood_size)
        ext_max.set_xdata(x_ext)
        ext_max.set_ydata(y_ext)
        flex_max.set_xdata(x_flex)
        flex_max.set_ydata(y_flex)
        contraction_values = [Z / It_ext_max * 100, Z1 / It_flex_max * 100, Z0 / It_cc_max * 100]

        for rect, h in zip(im, contraction_values):
            rect.set_height(h)
        global annotation0, annotation1, annotation2, cbar
        annotation0.remove()
        annotation1.remove()
        annotation2.remove()
        cbar = fig.colorbar(im3, cax=cbar_ax)
        cbar.set_label("mV", rotation=90)
        annotation0 = axs[0].annotate(
            str(round(contraction_values[0])),
            xy=(contraction_index[0], contraction_values[0]),
            ha="center", va="bottom",
        )
        annotation1 = axs[0].annotate(
            str(round(contraction_values[1])),
            xy=(contraction_index[1], contraction_values[1]),
            ha="center", va="bottom",
        )
        annotation2 = axs[0].annotate(
            str(round(contraction_values[2])),
            xy=(contraction_index[2], contraction_values[2]),
            ha="center", va="bottom",
        )
        text_time.set_text(f"time: {time_sec} seconds")
        text_action.set_text(current_action)
        fig.canvas.draw()

    ext_pp = activation_map(ext_pp, epoch)
    flex_pp = activation_map(flex_pp, epoch)

    emg_class = class_map(emg_class, epoch)
    valid_classes = np.array([i for i, v in enumerate(emg_class) if v.is_integer()])

    # --- Interactive plot setup ---

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

    plt.subplots_adjust(bottom=0.25)
    axs[1].set_title("EMG1 ")
    axs[2].set_title("EMG2")
    axs[0].set_title("Contraction Percentages")

    Z0, Z, Z1, Z2, Z3, time_sec, current_action = map_values(0, epoch)

    It_ext, It_ext_max = intensity_max(ext_pp)
    It_flex, It_flex_max = intensity_max(flex_pp)
    It_cc, It_cc_max = intensity_max(ext_pp + flex_pp)

    contraction_values = [Z / It_ext_max * 100, Z1 / It_flex_max * 100, Z0 / It_cc_max * 100]
    im = axs[0].bar(contraction_index, contraction_values)

    annotation0 = axs[0].annotate(
        str(round(contraction_values[0])),
        xy=(contraction_index[0], contraction_values[0]),
        ha="center", va="bottom",
    )
    annotation1 = axs[0].annotate(
        str(round(contraction_values[1])),
        xy=(contraction_index[1], contraction_values[1]),
        ha="center", va="bottom",
    )
    annotation2 = axs[0].annotate(
        str(round(contraction_values[2])),
        xy=(contraction_index[2], contraction_values[2]),
        ha="center", va="bottom",
    )

    axs[0].set_ylim([0, 100])

    typedraw = "none"
    im3 = axs[1].imshow(np.asarray(Z3), interpolation=typedraw)
    im4 = axs[2].imshow(np.asarray(Z2), interpolation=typedraw)
    im3.set_cmap("jet")
    im4.set_cmap("jet")
    text_time = plt.text(-10, 9, f"time: {time_sec} seconds", fontsize=10)
    text_action = plt.text(0, 9, current_action, fontsize=10)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label("mV", rotation=90)

    cbar_lim2 = [np.max(ext_pp), np.max(flex_pp)]
    threshold_max = max(cbar_lim2)
    neighborhood_size = 3

    im3.set_clim(vmin=0, vmax=np.max(cbar_lim2))
    im4.set_clim(vmin=0, vmax=np.max(cbar_lim2))
    x_ext, y_ext = local_maximum_pos(Z3, threshold_max * 0.05, neighborhood_size)
    x_flex, y_flex = local_maximum_pos(Z2, threshold_max * 0.05, neighborhood_size)

    (ext_max,) = axs[1].plot(x_ext, y_ext, "ko", label="local max")
    fig.legend(loc="upper right")
    (flex_max,) = axs[2].plot(x_flex, y_flex, "ko", label="local max")

    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    s_factor = Slider(ax_slide, "Time percentage", 0, 100, valinit=0, valstep=1)

    # --- Mean activation plot ---

    fig1, ax1 = plt.subplots(nrows=1, ncols=2)
    ax1[0].set_title("Extensor ")
    ax1[1].set_title("Flexor")
    typedraw = "none"

    im5 = ax1[0].imshow(mean_activation(ext_pp), interpolation=typedraw)
    im6 = ax1[1].imshow(mean_activation(flex_pp), interpolation=typedraw)
    cbar_limits = [np.max(mean_activation(ext_pp)), np.max(mean_activation(flex_pp))]
    im5.set_clim(vmin=0, vmax=np.max(cbar_limits))
    im6.set_clim(vmin=0, vmax=np.max(cbar_limits))
    im5.set_cmap("jet")
    im6.set_cmap("jet")
    fig1.subplots_adjust(right=0.8)
    cbar_ax2 = fig1.add_axes([0.85, 0.15, 0.05, 0.7])

    cbar2 = fig1.colorbar(im6, cax=cbar_ax2)
    cbar2.set_label("mV", rotation=90)
    plt.savefig("data.png")

    s_factor.on_changed(update)

    # --- Image features plot ---

    coords_harris, hog_image, canny_edges, resized_img, labels, cog, coordinates_max = image_features(
        mean_activation(ext_pp)
    )

    fig2, axs2 = plt.subplots(nrows=2, ncols=4, figsize=(12, 5))
    plt.axis("off")
    typedraw = "gaussian"
    axs2[0, 0].imshow(hog_image, cmap="jet", interpolation=typedraw)
    axs2[0, 0].set_ylabel("Extensor", fontsize=18)
    axs2[0, 1].imshow(resized_img, cmap="jet", interpolation=typedraw)
    axs2[0, 1].plot(
        coords_harris[:, 1], coords_harris[:, 0],
        color="black", marker="o", linestyle="None", markersize=6, label="Harris",
    )
    axs2[0, 1].plot(cog[0], cog[1], color="magenta", marker="o", linestyle="None", markersize=6, label="CoG")
    axs2[0, 1].legend(loc="upper right")
    axs2[0, 2].imshow(canny_edges, interpolation=typedraw, label="Canny edges")
    axs2[0, 2].plot(coordinates_max[:, 1], coordinates_max[:, 0], "r*", label="Local max")
    axs2[0, 2].legend(loc="upper right")
    axs2[0, 3].imshow(labels, cmap="jet")
    axs2[0, 1].set_title("Harris corners \n and CoG ")
    axs2[0, 0].set_title(" Histogram of \nOriented Gradients")
    axs2[0, 2].set_title("Canny edges and\n peak local max")
    axs2[0, 3].set_title("Mean shift\n features")

    coords_harris, hog_image, canny_edges, resized_img, labels, cog, coordinates_max = image_features(
        mean_activation(flex_pp)
    )

    axs2[1, 0].imshow(hog_image, cmap="jet", interpolation=typedraw)
    axs2[1, 0].set_ylabel("Flexor", fontsize=18)
    axs2[1, 1].imshow(resized_img, cmap="jet", interpolation=typedraw)
    axs2[1, 1].plot(
        coords_harris[:, 1], coords_harris[:, 0],
        color="black", marker="o", linestyle="None", markersize=6,
    )
    axs2[1, 1].plot(cog[0], cog[1], color="magenta", marker="o", linestyle="None", markersize=6)
    axs2[1, 2].imshow(canny_edges, interpolation=typedraw)
    axs2[1, 2].plot(coordinates_max[:, 1], coordinates_max[:, 0], "r*", label="Local max")
    axs2[1, 3].imshow(labels, cmap="jet")

    plt.show()
