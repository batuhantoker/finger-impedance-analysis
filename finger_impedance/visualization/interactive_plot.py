from matplotlib.widgets import Slider
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, lfilter
from skimage.transform import resize
from skimage.feature import hog, daisy
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_fast, peak_local_max, canny
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2 as cv
import pandas as pd

contraction_index = ['EMG1', 'EMG2', 'co-contraction']
epoch = 200


# Updating the plot
def update(val):
    current_i = s_factor.val
    global cbar_lim2, neighborhood_size, threshold_max
    Z0, Z, Z1, Z2, Z3, time_sec, current_action = map_values(round(current_i * len(ext_pp) / 100), epoch)
    typedraw = 'none'
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
    cbar.set_label('mV', rotation=90)
    annotation0 = axs[0].annotate(str(round(contraction_values[0])),
                                  xy=(contraction_index[0], contraction_values[0]), ha='center', va='bottom')
    annotation1 = axs[0].annotate(str(round(contraction_values[1])),
                                  xy=(contraction_index[1], contraction_values[1]), ha='center', va='bottom')
    annotation2 = axs[0].annotate(str(round(contraction_values[2])),
                                  xy=(contraction_index[2], contraction_values[2]), ha='center', va='bottom')

    text_time.set_text(f"time: {time_sec} seconds")
    text_action.set_text(current_action)

    # redrawing the figure
    fig.canvas.draw()


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities

    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    #edged = cv.Sobel(image, ddepth=cv.CV_64F , dx=1, dy=0)
    # return the edged image
    return edged


def image_features(data):
    resized_img = resize(data, (32, 32))
    resized_img = np.uint8((255 * (resized_img - np.min(resized_img)) / np.ptp(resized_img)).astype(int))
    coords_harris = corner_peaks(corner_harris(resized_img))
    coords_subpix = corner_subpix(resized_img, coords_harris, window_size=5)
    canny_edges = auto_canny((resized_img))  # canny(resized_img, sigma=2)  #
    fd, hog_image = hog(resized_img,  visualize=True) #orientations=5, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
    flat_image = np.reshape(resized_img, [-1, 1])
    # Estimate bandwidth
    bandwidth2 = estimate_bandwidth(flat_image, quantile=.1, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth2)
    ms.fit(flat_image)
    labels = np.reshape(ms.labels_, [32, 32])  # np.asarray(ms.labels_)
    cog = scipy.ndimage.center_of_mass(resized_img)
    image_max = scipy.ndimage.maximum_filter(resized_img, size=1, mode='constant')
    coordinates_max = peak_local_max(resized_img, min_distance=1,num_peaks=3)
    return coords_harris, hog_image, canny_edges, resized_img, labels, cog, coordinates_max


def data_reshape(data):
    data = np.reshape(data, (len(data), 8, 8))
    data = data.astype(np.float64)
    return data


# Maximum contraction for each channel
def mvc_calculator(ext, flex):
    mvc_ext = np.amax(ext, axis=0)
    mvc_flex = np.amax(flex, axis=0)
    # mvcc = np.amax(ext+flex, axis=0)
    return mvc_ext, mvc_flex  # , mvcc


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


# Intensity and maximum intensity at given time
def intensity_max(data):
    It = np.empty(len(data))
    for i in range(len(data)):
        It[i] = np.sum(data[i, :, :])
    It_max = np.amax(It)
    return It, It_max


def mean_activation(am):
    mean_activity = (np.mean(am, axis=0))
    return mean_activity


# load data
ext_raw = data_reshape(np.loadtxt('ext_raw.txt'))
ext_pp = data_reshape(np.loadtxt('ext_pp.txt'))

flex_raw = data_reshape(np.loadtxt('flex_raw.txt'))
flex_pp = data_reshape(np.loadtxt('flex_pp.txt'))

# ##Veronika
# ext_pp = data_reshape(np.loadtxt('emg1.txt'))
# flex_pp = data_reshape(np.loadtxt('emg2.txt'))


# [0:len(ext_pp),:,:]

emg_class = (np.loadtxt('emg_class.txt'))


def map_values(i, epoch):
    if emg_class[i * epoch] == 0:
        current_action = "rest"
    else:
        current_action = f"performing gesture {int(emg_class[i * epoch])}"  #
    Z0 = It_cc[i]
    Z = It_ext[i]
    Z1 = It_flex[i]
    Z2 = flex_pp[i, :, :]  # np.divide(flex_pp[i, :, :],mvc_flex)#
    Z3 = ext_pp[i, :, :]  # np.divide(ext_pp[i, :, :],mvc_ext)#
    time_sec = round(i * 0.00048828125 * epoch, 3)

    return Z0, Z, Z1, Z2, Z3, time_sec, current_action


def local_maximum_pos(data, threshold, neighborhood_size):
    data_max = scipy.ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = scipy.ndimage.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)  #
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


def activation_map(data, epoch):
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0:number_of_segments * epoch, :, :], number_of_segments)
    AM = np.empty([number_of_segments, data.shape[1], data.shape[2]])
    for i in range(number_of_segments):
        AM[i, :, :] = np.sqrt(np.mean(np.square(splitted_data[i]), axis=0))
    return AM
def class_map(data, epoch):
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0:number_of_segments * epoch], number_of_segments)
    class_value = np.empty([number_of_segments])
    for i in range(number_of_segments):
        class_value[i] = np.sqrt(np.mean(np.square(splitted_data[i])))
    return class_value

ext_pp = activation_map(ext_pp, epoch)
flex_pp = activation_map(flex_pp, epoch)

emg_class = class_map(emg_class,epoch)
valid_classes = np.array([i for i, v in enumerate(emg_class) if v.is_integer()])



# matplot subplot
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

plt.subplots_adjust(bottom=0.25)
axs[1].set_title('EMG1 ')
axs[2].set_title('EMG2')
axs[0].set_title('Contraction Percentages')
# axs[1, 1].set_title('Center of activation')
Z0, Z, Z1, Z2, Z3, time_sec, current_action = map_values(0, epoch)

im = axs[0].bar(contraction_index, contraction_values)

global annotation0, annotation1, annotation2, cbar
annotation0 = axs[0].annotate(str(round(contraction_values[0])), xy=(contraction_index[0], contraction_values[0]),
                              ha='center', va='bottom')
annotation1 = axs[0].annotate(str(round(contraction_values[1])), xy=(contraction_index[1], contraction_values[1]),
                              ha='center', va='bottom')
annotation2 = axs[0].annotate(str(round(contraction_values[2])), xy=(contraction_index[2], contraction_values[2]),
                              ha='center', va='bottom')

axs[0].set_ylim([0, 100])
# im2 = axs[1, 1].scatter((unravel_index(Z2.argmax(), Z2.shape))[0],(unravel_index(Z2.argmax(), Z2.shape))[1], c="blue")
# axs[1, 1].scatter((unravel_index(Z3.argmax(), Z3.shape))[0], (unravel_index(Z3.argmax(), Z3.shape))[1], c="red")

typedraw = 'none'
im3 = axs[1].imshow(np.asarray(Z3), interpolation=typedraw)
im4 = axs[2].imshow(np.asarray(Z2), interpolation=typedraw)
im3.set_cmap("jet")
im4.set_cmap("jet")
text_time = plt.text(-10, 9, f"time: {time_sec} seconds", fontsize=10)
text_action = plt.text(0, 9, current_action, fontsize=10)
# add space for colour bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
cbar = fig.colorbar(im3, cax=cbar_ax)
cbar.set_label('mV', rotation=90)
global cbar_lim2
cbar_lim2 = [np.max(ext_pp), np.max(flex_pp)]
im3.set_clim(vmin=0, vmax=np.max(cbar_lim2))
im4.set_clim(vmin=0, vmax=np.max(cbar_lim2))
x_ext, y_ext = local_maximum_pos(Z3, threshold_max * 0.05, neighborhood_size)
x_flex, y_flex = local_maximum_pos(Z2, threshold_max * 0.05, neighborhood_size)

ext_max, = axs[1].plot(x_ext, y_ext, 'ko', label="local max")
fig.legend(loc="upper right")
flex_max, = axs[2].plot(x_flex, y_flex, 'ko', label="local max")
# Defining the Slider button
# xposition, yposition, width and height
ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
# Properties of the slider
s_factor = Slider(ax_slide, 'Time percentage',
                  0, 100, valinit=0, valstep=1)

## Intensity plot
# fig1, ax1 = plt.subplots()
# ax1.plot(butter_lowpass_filter(It_ext/It_ext_max*100,10,2048,5),'r',label='extensor')
# ax1.plot(butter_lowpass_filter(It_flex/It_flex_max*100,10,2048,5),'b',label='flexor')
# ax1.plot(butter_lowpass_filter(It_cc/It_cc_max*100,10,2048,5),'g',label='co-contraction')
fig1, ax1 = plt.subplots(nrows=1, ncols=2)
ax1[0].set_title('Extensor ')
ax1[1].set_title('Flexor')
typedraw = 'none'

im5 = ax1[0].imshow(mean_activation(ext_pp), interpolation=typedraw)
im6 = ax1[1].imshow(mean_activation(flex_pp), interpolation=typedraw)
# ax1[0].set_xlim(0, 8)
# ax1[1].set_ylim(8, 0)
cbar_limits = [np.max(mean_activation(ext_pp)), np.max(mean_activation(flex_pp))]
im5.set_clim(vmin=0, vmax=np.max(cbar_limits))
im6.set_clim(vmin=0, vmax=np.max(cbar_limits))
im5.set_cmap("jet")
im6.set_cmap("jet")
fig1.subplots_adjust(right=0.8)
cbar_ax2 = fig1.add_axes([0.85, 0.15, 0.05, 0.7])

cbar2 = fig1.colorbar(im6, cax=cbar_ax2)
cbar2.set_label('mV', rotation=90)
# fig1.legend()
# ax1.set_title("Contraction percentages for movement 61 and 63")
# ax1.set_xlabel("")
plt.savefig('data.png')

# Calling the function "update" when the value of the slider is changed
s_factor.on_changed(update)

# print(flex_pp[6, :, :].shape)


coords_harris, hog_image, canny_edges, resized_img, labels, cog, coordinates_max = image_features(mean_activation(ext_pp))
dict_ext={'Harris_ext': coords_harris, 'HoG_ext': hog_image,'canny_ext':canny_edges,'mean_shift_ext':labels,'Cog_ext':cog,'max_cord_ext':coordinates_max}

fig2, axs2 = plt.subplots(nrows=2, ncols=4, figsize=(12, 5))
plt.axis("off")
typedraw = 'gaussian'
axs2[0, 0].imshow(hog_image, cmap="jet", interpolation=typedraw)
axs2[0, 0].set_ylabel('Extensor',fontsize=18)
axs2[0, 1].imshow(resized_img, cmap="jet", interpolation=typedraw)
axs2[0, 1].plot(coords_harris[:, 1], coords_harris[:, 0], color='black', marker='o', linestyle='None', markersize=6,
                label="Harris")
axs2[0, 1].plot(cog[0], cog[1], color='magenta', marker='o', linestyle='None', markersize=6, label="CoG")
axs2[0, 1].legend(loc="upper right")
axs2[0, 2].imshow(canny_edges, interpolation=typedraw, label="Canny edges")
axs2[0, 2].plot(coordinates_max[:, 1], coordinates_max[:, 0], 'r*', label="Local max")
axs2[0, 2].legend(loc="upper right")
axs2[0, 3].imshow(labels, cmap="jet")
axs2[0, 1].set_title('Harris corners \n and CoG ')
axs2[0, 0].set_title(' Histogram of \nOriented Gradients')
axs2[0, 2].set_title('Canny edges and\n peak local max')
axs2[0, 3].set_title('Mean shift\n features')
coords_harris, hog_image, canny_edges, resized_img, labels, cog, coordinates_max = image_features(mean_activation(flex_pp))

axs2[1, 0].imshow(hog_image, cmap="jet", interpolation=typedraw)
axs2[1, 0].set_ylabel('Flexor',fontsize=18)
axs2[1, 1].imshow(resized_img, cmap="jet", interpolation=typedraw)
axs2[1, 1].plot(coords_harris[:, 1], coords_harris[:, 0], color='black', marker='o', linestyle='None', markersize=6)
axs2[1, 1].plot(cog[0], cog[1], color='magenta', marker='o', linestyle='None', markersize=6)
axs2[1, 2].imshow(canny_edges, interpolation=typedraw)
axs2[1, 2].plot(coordinates_max[:, 1], coordinates_max[:, 0], 'r*', label="Local max")
axs2[1, 3].imshow(labels, cmap="jet")
# axs2[1,2].set_title('Harris and FAST corners')
# axs2[1,1].set_title('%i DAISY descriptors extracted:' % descs_num)
# axs2[1,0].set_title(' Histogram of Oriented Gradients')

#dict_flex={'Harris_flex': coords_harris, 'HoG_flex': hog_image,'canny_flex':canny_edges,'mean_shift_flex':labels,'Cog_flex':cog,'max_cord_flex':coordinates_max}
#z = dict(dict_flex, **dict_ext)
#dataset = pd.DataFrame.from_dict(z, orient='index').T


plt.show()

# def dataset_creation(i):
#     #coords_harris, hog_image, canny_edges, resized_img, labels, cog, coordinates_max = image_features(ext_pp[i, :, :])
#     #dict_ext = {'Harris_ext': coords_harris, 'HoG_ext': hog_image, 'canny_ext': canny_edges, 'mean_shift_ext': labels,
#     #            'Cog_ext': cog, 'max_cord_ext': coordinates_max}
#     resized_img=resize(ext_pp[i, :, :], (8, 8))
#     dict_ext = {'resized_img_ext':resized_img}
#     #coords_harris, hog_image, canny_edges, resized_img, labels, cog, coordinates_max = image_features(flex_pp[i, :, :])
#     #dict_flex = {'Harris_flex': coords_harris, 'HoG_flex': hog_image, 'canny_flex': canny_edges,
#      #            'mean_shift_flex': labels, 'Cog_flex': cog, 'max_cord_flex': coordinates_max}
#     resized_img = resize(flex_pp[i, :, :], (8, 8))
#     dict_flex = {'resized_img_flex': resized_img}
#     dict_target = {'movement_id': emg_class[i]}
#     z = dict(dict_flex, **dict_ext)
#     z2 = dict(z, **dict_target)
#     return z2
#
# diction=dataset_creation(0)
# df = pd.DataFrame.from_dict(diction, orient='index').T
# for i in range(1,len(valid_classes)): #
#     print(i)
#     valid_index=valid_classes[i]
#     diction = dataset_creation(valid_index)
#     dataset = pd.DataFrame.from_dict(diction, orient='index').T
#     df=pd.concat([df,dataset])
# df=df.reset_index(drop=True)
# print(df)
# print(df.shape)
# df.to_pickle("./data6.pkl")

