import numpy as np
import matplotlib.pyplot as plt
import pylab

from utils.data_utils import gen_noise, load_data


try:
    data_load = np.load("../data.npy", allow_pickle=True)
except FileNotFoundError:
    load_data()
    data_load = np.load("../data.npy", allow_pickle=True)

intensity = data_load[835][0]
labels = data_load[835][1]
input_size = len(intensity)
num_labels = len(labels)


def to_matrix(vector):
    pixels = np.zeros((35, 35))
    for i in range(35):
        pixels[i][0:34] = vector[i * 34: (i + 1) * 34]

    pixels = np.delete(pixels, 0, 0)
    pixels = np.delete(pixels, -1, axis=1)
    return pixels


signal_shot, signal_temp = gen_noise(
    signal=data_load[0][0],
    nb_photons=1e7,
    longueur_onde=1100,
    dim=input_size
)


pixels = to_matrix(intensity)
pixels_shotnoise = to_matrix(signal_shot)
pixels_tempnoise = to_matrix(signal_temp)


# _________________________________________ Plotting Data __________________________________________________

# plot intensity graphs in grayscale
fig, a = plt.subplots(1, 2)
a[0].imshow(pixels,  cmap=pylab.gray())
a[1].imshow(np.log10(pixels), cmap=pylab.gray())
plt.show()

fig, a = plt.subplots(1, 2)
a[0].imshow(pixels_shotnoise,  cmap=pylab.gray())
a[1].imshow(np.log10(pixels_shotnoise), cmap=pylab.gray())
plt.show()

fig, a = plt.subplots(1, 2)
a[0].imshow(pixels_tempnoise,  cmap=pylab.gray())
a[1].imshow(np.log10(pixels_tempnoise), cmap=pylab.gray())
plt.show()

# plot differentiation curves
i = 0

plt.style.use('seaborn-darkgrid')

fig1, ax1 = plt.subplots(2, 1, figsize=(5.7, 5.7))
fig2, ax2 = plt.subplots(2, 1, figsize=(6, 6))
fig3, ax3 = plt.subplots(2, 1, figsize=(6, 6))

for data in data_load:
    if data[1][i] == 1:

        signal_shot, signal_temp = gen_noise(
            signal=data[0],
            nb_photons=1e7,
            longueur_onde=1100,
            dim=input_size
        )

        pixels = to_matrix(data[0])
        pixels_shotnoise = to_matrix(signal_shot)
        pixels_tempnoise = to_matrix(signal_temp)

        linear_ccd = pixels[17:, 17]
        linear_ccd_shot = pixels_shotnoise[17:, 17]
        linear_ccd_temp = pixels_tempnoise[17:, 17]

        ax1[0].plot(linear_ccd, label="label" + str(i+1))
        ax1[1].plot(linear_ccd, label="label" + str(i+1))

        ax2[0].plot(linear_ccd_shot, label="label" + str(i+1))
        ax2[1].plot(linear_ccd_shot, label="label" + str(i+1))

        ax3[0].plot(linear_ccd_temp, label="label" + str(i+1))
        ax3[1].plot(linear_ccd_temp, label="label" + str(i+1))

        i += 1
        if i == 5:
            break

ax1[1].legend()
fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
fig1.suptitle("Curves without noise")
for ax in ax1.flat:
    ax.set(xlabel="Radius (mm)", ylabel="Percentage of detected photons (%)")
ax1[1].set_xlim(1, 3)

ax2[1].legend()
fig2.suptitle("Curves with Shotnoise")
for ax in ax2.flat:
    ax.set(xlabel="Radius (mm)", ylabel="Percentage of detected photons (%)")
ax2[1].set_xlim(1, 3)

ax3[1].legend()
fig3.suptitle("Curves with Shotnoise and Thermal noise")
for ax in ax3.flat:
    ax.set(xlabel="Radius (mm)", ylabel="Percentage of detected photons (%)")
ax3[1].set_xlim(3, 4)
plt.show()
